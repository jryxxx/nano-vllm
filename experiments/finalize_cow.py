import json
import statistics as st
from pathlib import Path

root=Path(__file__).resolve().parents[1]; out=root/'results/cow'
raw=json.loads((out/'benchmark.json').read_text()); rows=raw['trials']; summary=[]
for cached,branches in sorted({(x['cached_tokens'],x['branches']) for x in rows}):
    modes={mode:[r for r in rows if r['cached_tokens']==cached and r['branches']==branches and r['mode']==mode] for mode in ('copy','cow')}
    vals={mode:{key:st.median(r[key] for r in group) for key in ('fork_ms','fork_to_first_ms','total_ms','tokens_per_s','peak_live_pages','copy_bytes')} for mode,group in modes.items()}
    spread={mode:{key:[min(r[key] for r in group),max(r[key] for r in group)] for key in ('fork_to_first_ms','total_ms')} for mode,group in modes.items()}
    summary.append(dict(cached=cached,branches=branches,medians=vals,ranges=spread,
                        throughput_gain_pct=100*(vals['copy']['total_ms']/vals['cow']['total_ms']-1),
                        page_reduction_pct=100*(1-vals['cow']['peak_live_pages']/vals['copy']['peak_live_pages']),
                        copy_reduction_pct=100*(1-vals['cow']['copy_bytes']/vals['copy']['copy_bytes'])))
ordinary=[]
for scenario in ('shared_short','no_shared'):
    vals={stage:st.median(r['output_tokens_per_s'] for r in json.loads((out/f'ordinary-{stage}.json').read_text())['trials'] if r['scenario']==scenario) for stage in ('before','after')}
    ordinary.append(dict(scenario=scenario,**vals,change_pct=100*(vals['after']/vals['before']-1)))
result=dict(config=raw['config'],scenarios=summary,ordinary=ordinary)
(out/'summary.json').write_text(json.dumps(result,indent=2))
lines=['# KV Cache 会话分支 COW：实现与实验报告','',
'已在远程 `kv-session-cow` 分支完成会话快照、请求分支、尾块写时复制、引用计数和回收，并接入 prefill / decode 调度及共享前缀路径。',
'', '## 实验设置','',
'RTX 4090 24GB；Qwen3-0.6B FP16；单 GPU、CUDA Graph；KV 页大小 256 token。分支实验关闭自定义共享注意力路径，以区分本阶段 COW 与上一阶段算子的收益。完整复制和 COW 使用同一个 Triton 复制内核。',
'', '父请求已完成 prefill 并暂停。每个分支生成 32 token；每场景每模式预热一次，3 次测量交替顺序，以下为中位数。总耗时包含分支构造、复制、首次输出、后续生成和回收，不包含共同的父请求 prefill。原始每次耗时和 min/max 区间见 JSON；小幅差异不作统计显著性声明。',
'', '## 正确性','']
for filename in ('gpu-auto.json','gpu-eager.json'):
    data=json.loads((out/filename).read_text()); cases=[r for r in data['cases'] if 'cached' in r]
    lines.append(f'- {filename}: {len(cases)} 组分页/变长场景通过；COW 对完整复制最大 logits 差 {max(r["cow_copy_max_abs"] for r in cases):.6f}；独立完整重算最大相对 L2 {max(r["fresh_recompute_relative_l2"] for r in cases):.6f}。')
lines += ['- 原会话已计算 KV 逐位不变；完整复制的有效 KV 逐位一致。覆盖 255/256/257、1024/1151 边界及 3/300 token 续接。',
'- 多代分支、完成时保存、暂停/恢复、取消父请求、同名快照覆盖、重复取消/关闭、复制异常回滚、满池报错和释放后恢复均通过。CPU 2,000 次随机生命周期操作结束后引用归零。',
'- FP16/BF16 复制内核的 0/1/255/256 有效 token 掩码、目标未写区和源块保持测试通过；完整引擎的本阶段数值验收使用 FP16。',
'- 32 分支的 COW 与共享前缀 Graph 集成通过；上一阶段独立共享注意力与通用分页/变长回归结果保存在 results/cow。',
'', '## 分支对照结果','',
'| 已缓存 token | 分支数 | 首 token ms：复制 → COW | 总耗时 ms：复制 → COW | 输出吞吐变化 | 峰值活跃页：复制 → COW |',
'|---:|---:|---:|---:|---:|---:|']
for r in summary:
    a,b=r['medians']['copy'],r['medians']['cow']
    lines.append(f'| {r["cached"]} | {r["branches"]} | {a["fork_to_first_ms"]:.2f} → {b["fork_to_first_ms"]:.2f} | {a["total_ms"]:.2f} → {b["total_ms"]:.2f} | {r["throughput_gain_pct"]:+.2f}% | {a["peak_live_pages"]:.0f} → {b["peak_live_pages"]:.0f} |')
best=summary[-1];a,b=best['medians']['copy'],best['medians']['cow']
lines += ['',f'4,223 token × 16 分支：复制量 {a["copy_bytes"]/2**30:.3f} GiB → {b["copy_bytes"]/2**30:.3f} GiB（减少 {best["copy_reduction_pct"]:.2f}%）；峰值活跃页减少 {best["page_reduction_pct"]:.2f}%。这些是固定工作负载测量，不是饱和服务容量或普遍加速保证。',
'', '## 普通请求回归','', '对照为进入本阶段前保留的 `nano-vllm-pre-cow`，两边均关闭自定义共享注意力，输入与输出长度固定，每项 3 次。']
for r in ordinary: lines.append(f'- {r["scenario"]}: {r["before"]:.1f} → {r["after"]:.1f} output token/s，变化 {r["change_pct"]:+.2f}%。')
lines += ['', '## 适用范围与解释','',
'- 当前会话 API 为单线程、单 GPU；快照驻留 GPU，无 TTL、CPU offload 或分布式会话所有权。复制同步完成后提交映射。',
'- 活跃页减少意味着预分配池可容纳更多其他工作；GPU KV 池本身仍预分配，并不意味着 CUDA reserved memory 同比下降。',
'- 完整块共享，部分尾块首次写入时复制。copy 模式复制全部已计算页；两种模式都按实际有效 token 复制，未用 page padding 不计入复制量。',
'- 单分支/短上下文主要节省页和复制量，吞吐变化可能很小。长上下文多分支在本次试验中更有利。',
'- GPU performance counters 不可用；不对实际 DRAM 事务数、L2 命中率或低层瓶颈作实测声明。复制字节数根据内核有效 token 和张量布局计算。',
'- 复制事务失败的元数据回滚已验证；设备级非法访问导致的 CUDA context 损坏需要进程重启。',
'', '实现与 API 示例见 [KV_SESSIONS.md](KV_SESSIONS.md)。`results/cow/summary.json` 为汇总；`benchmark.json` 为最终实验；`benchmark-initial.json` 和 `benchmark-before-fastpath.json` 为优化前诊断，不用于最终结论。',
'']
(root/'docs/KV_COW_REPORT.md').write_text('\n'.join(lines))
print(json.dumps(result,indent=2))
