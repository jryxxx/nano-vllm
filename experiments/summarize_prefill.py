import json,statistics as st
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/prefill'
raw=json.loads((OUT/'replay.json').read_text());rows=raw['trials'];variants=list(raw['config']['variants'])
scenarios=['mixed','overload','cold_only','two_hot_groups','short_hot_groups'];summary={}
for scenario in scenarios:
    summary[scenario]={}
    for variant in variants:
        group=[r for r in rows if r['scenario']==scenario and r['variant']==variant];assert len(group)==5
        assert all(r['unfinished']==0 for r in group)
        metrics=['elapsed_s','output_tokens_per_s','goodput','good','offered','accepted','rejected','peak_live_pages']
        med={k:st.median(r[k] for r in group) for k in metrics};ranges={k:[min(r[k] for r in group),max(r[k] for r in group)] for k in metrics}
        cohorts={}
        for kind in ('hot','cold'):
            if kind not in group[0]['cohorts']:continue
            cohorts[kind]={key:{p:st.median(r['cohorts'][kind][key][p] for r in group) for p in ('50','95','99','100')}
                           for key in ('ttft_ms','request_max_itl_ms')}
            cohorts[kind].update({key:st.median(r['cohorts'][kind][key] for r in group) for key in ('good','offered','rejected')})
        decisions=Counter();routes=Counter()
        for r in group:decisions.update(r['scheduler_metrics']);routes.update(r['routes'])
        prefill_counts=[sum(s['work']>0 for s in r['steps']) for r in group]
        summary[scenario][variant]=dict(median=med,ranges=ranges,cohorts=cohorts,metrics=dict(decisions),routes=dict(routes),
            median_prefill_steps=st.median(prefill_counts),
            engine_prefill_prediction_mae_ms=decisions.get('prefill_prediction_abs_error_us',0)/1000/max(1,decisions.get('prefill_prediction_count',0)))
model=json.loads((OUT/'model-validation.json').read_text());audit=json.loads((OUT/'fresh-audit.json').read_text())
gpu=json.loads((OUT/'gpu-validation.json').read_text());assert gpu['passed']
checks=[r for c in gpu['cases'] for r in c['checks']]
quality=dict(request_token_comparisons=len(checks),max_relative_l2=max(r['relative_l2'] for r in checks),max_kl=max(r['max_kl'] for r in checks))
changes={}
for scenario in scenarios:
    old=summary[scenario]['fixed_v1'];new=summary[scenario]['joint']
    change=dict(throughput_pct=100*(new['median']['output_tokens_per_s']/old['median']['output_tokens_per_s']-1),
                goodput_pct=100*(new['median']['goodput']/old['median']['goodput']-1))
    if 'cold' in old['cohorts']:
        change['cold_ttft_p95_reduction_pct']=100*(1-new['cohorts']['cold']['ttft_ms']['95']/old['cohorts']['cold']['ttft_ms']['95'])
    changes[scenario]=change
result=dict(config=raw['config'],scenarios=summary,changes_vs_fixed_v1=changes,quality=quality,
            holdout_mape=model['mape'],fresh_audit_mape=audit['mape'],fresh_audit_max_error=audit['max_relative_error'])
(OUT/'summary.json').write_text(json.dumps(result,indent=2))
lines=['# Prefill 成本模型与联合调度：对照实验报告','',
'本阶段在上一版成本调度上加入可校准的 Prefill 成本模型、批次/分块选择，以及为 Prefill 腾出时间的 Decode 尾批补充服务。新模式显式启用，默认 baseline 不变。',
'', '## 实现与成本模型验证','',
'- 模型先拟合固定开销、query 数、query × cached-context/因果注意力工作量、缓存路径和 batch 特征的非负仿射骨架；精确标定形状使用测量中位数，未见形状使用邻近训练形状的残差校正。',
'- 共 51 个训练形状，每形状预热 2 次、测量 5 次；覆盖总 query ≤ 8192、batch ≤ 8、已有缓存 ≤ 3072 token。超出范围或缺少可信 Decode 成本时回退，不外推性能承诺。',
f'- 仿射骨架单独在 8 个留出形状上的 MAPE 为 16.09%；加入训练表局部校正后为 {100*model["mape"]:.2f}%。另测 6 个未加入模型的新形状，MAPE {100*audit["mape"]:.2f}%，最大相对误差 {100*audit["max_relative_error"]:.2f}%。',
'- 旧版按每 token 平均耗时缩 chunk 的做法已不用。新的候选计划使用完整 Prefill 路径预测，并为后续 oldest-first Decode 轮次预留时间。预测乘 1.10 安全系数，另留 2 ms 计划开销；70 ms 为软服务预算，不是硬截止时间。',
'- 联合阶段策略在预算不足时先补充 Decode；若只有上一批未覆盖的尾部请求较旧，则只刷新这部分请求，避免反复执行已经得到服务的整批。等待工作仍受最大连续 Decode 次数保护，强制推进和预测失败均计数。',
'', '## 对照协议','',
'RTX 4090、Qwen3-0.6B FP16、CUDA Graph、同一预分配 KV 池、attention=auto。沿用上一阶段的自然语言格式合成文档、预定开环到达、输出长度、队列上限 96 和 SLO：TTFT ≤ 500 ms、请求内最大 ITL ≤ 75 ms。',
'',
'五种策略 × 五种负载 × 五次正式重复，共 125 次；每组另做一次预热不计入，策略顺序轮转。TTFT 从预定到达时刻计。父会话准备在计时外，客户端 fork/冷请求仅在到达时注入。goodput 为符合两项 SLO 的完成请求数除以有限请求队列处理完成时间。不是稳态容量或生产 P99 认证。',
'',
'策略：baseline=原始 Prefill 优先；fixed_v1=上一版固定 1024 chunk；fixed_large=仅改成固定 4096 chunk；model_only=模型预算＋旧阶段规则；joint=模型预算＋预测阶段选择＋Decode 尾批刷新。所有公平策略使用相同 Decode 成本模型和其他参数。',
'', '## 相对上一版 fixed_v1 的主要改善','',
'| 场景 | 输出吞吐 tok/s：旧 → 新 | 变化 | 冷 TTFT p95 ms：旧 → 新 | 降低 | goodput req/s：旧 → 新 |',
'|---|---:|---:|---:|---:|---:|']
for scenario in ('mixed','overload','cold_only'):
    a,b=summary[scenario]['fixed_v1'],summary[scenario]['joint'];c=changes[scenario]
    lines.append(f'| {scenario} | {a["median"]["output_tokens_per_s"]:.1f} → {b["median"]["output_tokens_per_s"]:.1f} | {c["throughput_pct"]:+.2f}% | {a["cohorts"]["cold"]["ttft_ms"]["95"]:.1f} → {b["cohorts"]["cold"]["ttft_ms"]["95"]:.1f} | {c["cold_ttft_p95_reduction_pct"]:.2f}% | {a["median"]["goodput"]:.2f} → {b["median"]["goodput"]:.2f} |')
lines += ['', '表中数值均为 5 次运行的中位数，分位数为每次运行内计算后取中位数；范围和逐请求数据保留在 JSON。改善参照上一版公平成本调度，不等同于对原始吞吐优先基线的全面改善。',
'', '## 完整消融与代价','',
'| 场景 | 策略 | 吞吐 tok/s | 冷 TTFT p95 ms | 热请求 max-ITL 的 p95 ms | goodput req/s | 接纳 / 拒绝 |',
'|---|---|---:|---:|---:|---:|---:|']
for scenario in ('mixed','overload','cold_only'):
    for variant in variants:
        d=summary[scenario][variant];m=d['median'];hot=d['cohorts'].get('hot')
        hot_text=f'{hot["request_max_itl_ms"]["95"]:.2f}' if hot else '—'
        lines.append(f'| {scenario} | {variant} | {m["output_tokens_per_s"]:.1f} | {d["cohorts"]["cold"]["ttft_ms"]["95"]:.1f} | {hot_text} | {m["goodput"]:.2f} | {m["accepted"]:g} / {m["rejected"]:g} |')
lines += ['',
'- fixed_large 在混合负载中吞吐更高、冷请求首 token 更快，但热点输出间隔超过固定 75 ms 标准；因此不能只靠扩大 chunk 宣称问题解决。',
'- model_only 能减少 Prefill 步数，但旧阶段规则仍会在没有余量时插入 Prefill。joint 通过阶段选择和尾批刷新改善这项取舍。',
'- joint 在过载下仍有较长冷 TTFT 和不达标请求；它改善的是上一版取舍，并未实现所有请求同时满足 SLO。',
'- 更早完成可能在到达流量尚未结束时重新开放队列槽，所以过载下各变体接纳数量可能不同。表内同时报告接纳与拒绝；没有隐藏被拒绝或未完成的请求。所有已接纳请求均完成，最终引用归零。',
'', '## 无 Prefill 场景回归','',
'| 场景 | fixed_v1 tok/s | joint tok/s | 变化 |',
'|---|---:|---:|---:|']
for scenario in ('two_hot_groups','short_hot_groups'):
    a=summary[scenario]['fixed_v1']['median']['output_tokens_per_s'];b=summary[scenario]['joint']['median']['output_tokens_per_s']
    lines.append(f'| {scenario} | {a:.1f} | {b:.1f} | {(b/a-1)*100:+.2f}% |')
lines += ['', '## 正确性','',
f'- 五种策略共 {quality["request_token_comparisons"]} 个逐请求/token 数值对齐点，覆盖不同 Prefill 分块、变长 batch、已缓存前缀、COW 分支和超出模型范围的回退。最大相对 L2 {quality["max_relative_l2"]:.6f}，最大 KL {quality["max_kl"]:.6f}，通过原有 0.005 / 0.001 门槛。',
'- COW 父会话有效尾部保持逐位不变；所有验证结束后 KV 引用归零。CPU 覆盖模型未知/越界回退、过期 Decode 优先、尾批刷新和多请求 Prefill 组批。',
'- 原调度回归、原 COW GPU 验证、37 个分页用例及 8 次动态 Graph replay 通过。',
'', '## 范围与复现','',
'- 成本表只在本次 GPU、模型和 dtype 上标定；不同设备和模型需重新校准。FP16 小模型实验不能替代更大模型或真实生产轨迹验证。',
'- 不使用 GPU Performance Counters，因此不把吞吐变化解释成已实测的 DRAM/L2 变化。',
'- 缓存淘汰、CPU 分层、跨模型调度与 BF16 共享引擎路径不在本阶段。',
'- `pilot.json`、`pilot-refresh.json` 为开发诊断，最终结论仅来自 `replay.json`。',
'', '实现与运行命令见 [PREFILL_MODEL.md](PREFILL_MODEL.md)。原始数据位于 `results/prefill/`：`cost-profile.json`、`model-validation.json`、`fresh-audit.json`、`gpu-validation.json`、`replay.json`、`summary.json`。','']
(ROOT/'docs/PREFILL_REPORT.md').write_text('\n'.join(lines))
print(json.dumps(dict(changes=changes,quality=quality,holdout_mape=model['mape'],fresh_audit_mape=audit['mape']),indent=2))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');names=['mixed','overload','cold_only'];show=['fixed_v1','fixed_large','model_only','joint'];colors=['#7c8796','#ef9645','#4d8cc7','#249c83'];x=np.arange(3);width=.19
for j,(variant,color) in enumerate(zip(show,colors)):
    ds=[summary[n][variant] for n in names];offset=(j-1.5)*width
    values=[[d['median']['output_tokens_per_s'] for d in ds],
            [d['cohorts']['cold']['ttft_ms']['95'] for d in ds],
            [d['median']['goodput'] for d in ds],
            [d['cohorts'].get('hot',d['cohorts']['cold'])['request_max_itl_ms']['95'] for d in ds]]
    for ax,v in zip(axes.flat,values):ax.bar(x+offset,v,width,label=variant,color=color)
for ax,title,ylabel in zip(axes.flat,['Output throughput','Cold-request TTFT p95','SLO goodput','Request maximum ITL p95*'],['output tokens/s','ms (lower is better)','requests/s','ms (lower is better)']):
    ax.set_xticks(x,names);ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[1,1].axhline(75,color='#a73535',linestyle='--',label='75 ms SLO')
axes[0,0].legend(fontsize=9);axes[1,1].legend(fontsize=9)
fig.suptitle('Prefill cost model and joint scheduling — median of 5 repeats\n*Hot requests when present; cold requests for cold_only',fontsize=13)
fig.savefig(OUT/'prefill-tradeoffs.png',dpi=170);plt.close(fig)
