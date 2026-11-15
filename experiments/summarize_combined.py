import json
import statistics as st
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/combined'
raw=json.loads((OUT/'benchmark.json').read_text()); rows=raw['trials']
variants=list(raw['config']['variants']); scenarios=list(dict.fromkeys(r['scenario'] for r in rows))
metrics=['parent_ms','fork_ms','branch_first_ms','cold_first_ms','branch_total_ms','cold_total_ms',
         'cold_tokens_per_s','branch_tokens_per_s','peak_live_pages','copy_bytes','planning_ms','live_after_fork']
summary={}
for scenario in scenarios:
    summary[scenario]={}
    for variant in variants:
        group=[r for r in rows if r['scenario']==scenario and r['variant']==variant]
        assert len(group)==raw['config']['repeats']
        med={m:st.median(r[m] for r in group) for m in metrics}
        ranges={m:[min(r[m] for r in group),max(r[m] for r in group)] for m in metrics}
        routes=Counter()
        for r in group:routes.update(r['routes']);assert r['final_live_pages']==0 and r['preemptions']==0
        summary[scenario][variant]=dict(median=med,ranges=ranges,routes=dict(routes),trials=len(group))
    base=summary[scenario]['copy_off']['median']
    for variant in variants:
        v=summary[scenario][variant]
        v['cold_speedup']=base['cold_total_ms']/v['median']['cold_total_ms']
        v['branch_speedup']=base['branch_total_ms']/v['median']['branch_total_ms']
        # Pair by workload seed/repeat; retain range instead of claiming significance.
        pairs=[]
        for repeat in range(raw['config']['repeats']):
            a=next(r for r in rows if r['scenario']==scenario and r['variant']=='copy_off' and r['repeat']==repeat)
            b=next(r for r in rows if r['scenario']==scenario and r['variant']==variant and r['repeat']==repeat)
            pairs.append(a['cold_total_ms']/b['cold_total_ms'])
        v['paired_cold_speedup']=pairs
        v['paired_cold_range']=[min(pairs),max(pairs)]
validation=json.loads((OUT/'validation.json').read_text()); assert validation['passed']
checks=[x for case in validation['cases'] for x in case['checks']]
quality=dict(cases=len(validation['cases']),steps=len(checks),max_relative_l2=max(x['relative_l2'] for x in checks),
             max_kl=max(x['max_kl'] for x in checks),max_abs=max(x['max_abs'] for x in checks),
             argmax_matches=sum(x['argmax_matches'] for x in checks),rows=sum(x['rows'] for x in checks))
report=dict(config=raw['config'],scenarios=summary,quality=quality)
(OUT/'summary.json').write_text(json.dumps(report,indent=2))

lines=['# 会话 COW × 共享前缀注意力：端到端对照与消融','',
'本阶段复用已验证的远程引擎，通过同一个配置对象切换现有 attention planner，未修改生产引擎的选择阈值。新增可复现实验和组合正确性测试。',
'', '## 六组定义与可识别性','',
'| 实验组 | 完整前缀物理页 | 部分尾页 | 注意力 |',
'|---|---|---|---|',
'| copy_off | 每分支复制 | 分支时复制 | 原分页注意力 |',
'| copy_auto | 每分支复制 | 分支时复制 | 开启 auto，因物理前缀不同回退 |',
'| cow_off | 共享 | 首次写入时复制 | 原分页注意力 |',
'| cow_auto | 共享 | 首次写入时复制 | auto 选择共享前缀注意力 |',
'| eager_tail_off | 共享 | 分支时立即物化私有尾页 | 原分页注意力 |',
'| eager_tail_auto | 共享 | 分支时立即物化私有尾页 | auto 选择共享前缀注意力 |',
'',
'当前共享注意力要求各请求的物理前缀页 ID 相同。完整复制 + auto 因此只是负对照，不能称为“共享注意力单独生效”。eager_tail 复用同一个 COW 复制事务，只把执行时机前移；它是控制实验中的“完整前缀共享 + 立即复制尾块”，不是新增的公共缓存后端。',
'',
'可归因对比：copy_off → eager_tail_off 衡量完整前缀共享及其伴随的数据复制/访存变化；eager_tail_off → eager_tail_auto 衡量同样页面布局上的共享注意力；eager_tail → cow 衡量尾块延迟物化；cow_off → cow_auto 衡量共享注意力在会话 COW 上的增量。不能把各项加速百分比直接相加。',
'', '## 方法','',
f'RTX 4090，Qwen3-0.6B FP16；单 GPU、CUDA Graph；256 token/页；所有组使用同一预分配池（{raw["config"]["pool_pages"]} 页），普通和共享 Graph 均提前捕获。每场景每组 1 次预热不计入，{raw["config"]["repeats"]} 次测量；六组轮转顺序，每组在各次序位置出现一次。',
'',
'输入是固定长度的合成 token 负载。每次测量重建空的逻辑块管理器，固定同一 repeat 的 prompt、pending token、随机种子和生成长度。冷会话总耗时包含父请求 prefill/一次输出、暂停、分支构造/复制、子请求生成和父请求回收；模型加载/编译/Graph 捕获排除。分支耗时从父请求完成后开始，包含首次写入 COW。吞吐分子仅计子请求输出 token，父请求准备成本在分母中。',
'',
'真实采样参与计时，ignore_eos 固定输出长度；不同注意力数值路径可能产生不同 token，因此另外使用完全相同的 teacher-forced 历史做数值验证。每轮记录路由计数、复制量、峰值活跃页、逐步耗时、结束引用和抢占次数。所有计时样本均无抢占、最终引用归零。小幅差异不作显著性结论；原始数据保留每次测量和配对范围。',
'', '## 冷会话端到端结果','',
'| 场景 | copy_off ms | cow_off ms | cow_auto ms | 组合相对复制吞吐 | 注意力相对 COW 增量 |',
'|---|---:|---:|---:|---:|---:|']
for name in scenarios:
    d=summary[name];a=d['copy_off']['median'];c=d['cow_off']['median'];both=d['cow_auto']['median']
    lines.append(f'| {name} | {a["cold_total_ms"]:.2f} | {c["cold_total_ms"]:.2f} | {both["cold_total_ms"]:.2f} | {(a["cold_total_ms"]/both["cold_total_ms"]-1)*100:+.2f}% | {(c["cold_total_ms"]/both["cold_total_ms"]-1)*100:+.2f}% |')
lines += ['', '场景：short_boundary=256 token/32 分支/32 输出；medium=1151/32/64；long=4223/32/64；small_batch=4223/8/64；shrinking=1151/33、输出 16/32/48 混合；two_groups=两个独立父会话各 1151 token、各 16 分支/64 输出；cancel_before_write=4223/32，首次写入前取消 24 个，剩余 8 个输出 64 token。',
'', '## 长上下文完整消融（4223 token × 32 分支）','',
'| 组 | 冷会话总 ms | 分支总 ms | 分支到首 token ms | 峰值活跃页 | 复制 GiB | 共享 step / 全部 decode step |',
'|---|---:|---:|---:|---:|---:|---:|']
for variant in variants:
    d=summary['long'][variant];m=d['median']; routes=d['routes']
    lines.append(f'| {variant} | {m["cold_total_ms"]:.2f} | {m["branch_total_ms"]:.2f} | {m["branch_first_ms"]:.2f} | {m["peak_live_pages"]:.0f} | {m["copy_bytes"]/2**30:.3f} | {routes.get("shared",0)} / {sum(routes.values())} |')
lines += ['', '## 尾块延迟物化：首次写入前取消 24/32 分支','',
'| 组 | 分支构造 ms | 分支到首 token ms | 冷会话总 ms | 复制 GiB | 峰值活跃页 |',
'|---|---:|---:|---:|---:|---:|']
for variant in ('copy_off','eager_tail_off','cow_off'):
    m=summary['cancel_before_write'][variant]['median']
    lines.append(f'| {variant} | {m["fork_ms"]:.2f} | {m["branch_first_ms"]:.2f} | {m["cold_total_ms"]:.2f} | {m["copy_bytes"]/2**30:.3f} | {m["peak_live_pages"]:.0f} |')
lines += ['', '该场景用于隔离按需复制的实际价值：cow 不为取消的分支复制尾页，eager_tail 已支付其复制成本。其余立即开始解码的场景，两者应复制相同数量的尾页；构造耗时减少不等于首 token 或总耗时必然同比改善。',
'', '## 路由与正确性','',
f'- {quality["cases"]} 组组合验证、{quality["steps"]} 个模型 step，涵盖长上下文 64 token 连续生成、33→22→11 缩批（16/32/48 输出）和 0/1/17/257 token 变长续接。全程使用原注意力的实验组与完整复制基线 logits 逐位一致。',
f'- 包含共享注意力及缩批后回退的路径：最大相对 L2 {quality["max_relative_l2"]:.6f}，最大 KL {quality["max_kl"]:.6f}，最大绝对 logits 差 {quality["max_abs"]:.6f}；argmax {quality["argmax_matches"]}/{quality["rows"]} 一致。沿用 L2 < 0.005、KL < 0.001 的门槛，不声称组合路径位级等价。',
'- 缩批回退时，KV 历史已包含此前共享注意力的舍入差异，因此仍按整体 L2/KL 门槛验证，不要求回退 step 恢复位级一致。',
'- medium/long 的 cow_auto 与 eager_tail_auto 实际命中共享路径；copy_auto 因 no_common_prefix 回退。short_boundary、small_batch、two_groups 分别验证短前缀、小批量和无共同前缀回退。shrinking 在降至 32 以下后回退；取消场景只剩 8 个活跃分支，也回退。',
'', '## 结论边界','',
'- 物理前缀共享是当前共享注意力算子的前提，二者不是完全独立的开关。页面共享的收益不应全部归因于延迟尾块 COW。',
'- 此处是固定并发与输出长度的端到端对照，不是开放环到达流量、生产 P99 SLA 或最大饱和容量测试。',
'- 活跃页数是预分配池中的占用；所有组的 GPU 池大小一致，不等于 CUDA reserved memory 同比下降。复制字节按有效 token 的单份 KV 数据量计算，不是硬件 DRAM 事务计数。',
'- GPU Performance Counters 未启用，不对缓存命中率或硬件访存量作实测归因。',
'', '## 产物与复现','',
'`experiments/bench_combined.py`：六组计时实验；`tests/validate_combined.py`：teacher-forced 组合验证；`results/combined/benchmark.json`：原始计时；`validation.json`：逐步数值；`summary.json`：中位数、范围及路由计数。',
'', '```bash',
'bash /root/autodl-tmp/shared-context/run_cow_env.sh python experiments/bench_combined.py \\',
'  --model /root/autodl-tmp/shared-context/models/Qwen3-0.6B-fp16 \\',
'  --output results/combined/benchmark.json',
'bash /root/autodl-tmp/shared-context/run_cow_env.sh python tests/validate_combined.py \\',
'  --model /root/autodl-tmp/shared-context/models/Qwen3-0.6B-fp16 \\',
'  --output results/combined/validation.json',
'```','']
(ROOT/'docs/COMBINED_ABLATION.md').write_text('\n'.join(lines))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
matrix=np.array([[summary[s][v]['cold_speedup'] for v in variants] for s in scenarios])
fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
im=ax.imshow(matrix,cmap='YlGnBu',vmin=.9,vmax=max(2.,matrix.max()))
ax.set_xticks(range(len(variants)),variants,rotation=20,ha='right')
ax.set_yticks(range(len(scenarios)),scenarios)
for i in range(len(scenarios)):
    for j in range(len(variants)):
        ax.text(j,i,f'{matrix[i,j]:.2f}x',ha='center',va='center',color='white' if matrix[i,j]>1.9 else 'black')
ax.set_title('Cold session throughput relative to full-copy / attention-off\nParent prefill + fork + generation + reclamation; median of 6 trials')
fig.colorbar(im,ax=ax,label='Throughput ratio (higher is better)')
fig.savefig(OUT/'cold-speedup.png',dpi=180);plt.close(fig)
print(json.dumps(dict(quality=quality,long=summary['long']),indent=2))
