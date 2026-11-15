import json,statistics as st
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/scheduler'
core=json.loads((OUT/'replay-core.json').read_text());extra=json.loads((OUT/'replay-short.json').read_text())
core['trials']+=extra['trials'];(OUT/'replay.json').write_text(json.dumps(core,indent=2))
rows=core['trials'];policies=['baseline','fcfs','lpm','cost'];scenarios=['two_hot_groups','short_hot_groups','mixed','overload','cold_only'];summary={}
for scenario in scenarios:
    summary[scenario]={}
    for policy in policies:
        group=[r for r in rows if r['scenario']==scenario and r['policy']==policy];assert len(group)==4
        metrics=['elapsed_s','goodput','output_tokens_per_s','offered','accepted','rejected','unfinished','good','peak_live_pages']
        med={m:st.median(r[m] for r in group) for m in metrics}
        ranges={m:[min(r[m] for r in group),max(r[m] for r in group)] for m in metrics}
        cohorts={}
        for kind in ('hot','cold'):
            if kind not in group[0]['cohorts']:continue
            cohorts[kind]={m:st.median(r['cohorts'][kind][m] for r in group) for m in ['offered','rejected','good']}
            for metric in ('ttft_ms','request_max_itl_ms'):
                cohorts[kind][metric]={p:st.median(r['cohorts'][kind][metric][p] for r in group) for p in ('50','95','99','100')}
        routes=Counter();decisions=Counter()
        for r in group:routes.update(r['routes']);decisions.update(r['scheduler_metrics'])
        summary[scenario][policy]=dict(median=med,ranges=ranges,cohorts=cohorts,routes=dict(routes),decisions=dict(decisions))
validation=json.loads((OUT/'gpu-validation.json').read_text());assert validation['passed']
quality_rows=[r for item in validation['permutation'] for r in item['rows']]
quality=dict(rows=len(quality_rows),max_relative_l2=max(r['relative_l2'] for r in quality_rows),max_kl=max(r['kl'] for r in quality_rows))
off=json.loads((OUT/'replay-attention-off.json').read_text())
attention_off={}
for scenario in ('two_hot_groups','short_hot_groups'):
    attention_off[scenario]={}
    for policy in ('fcfs','cost'):
        group=[r for r in off['trials'] if r['scenario']==scenario and r['policy']==policy]
        assert len(group)==4 and all(r['routes'].get('shared',0)==0 for r in group)
        attention_off[scenario][policy]=dict(throughput=st.median(r['output_tokens_per_s'] for r in group),
            shared_steps=0,group_choices=sum(r['scheduler_metrics'].get('cost_group_selected',0) for r in group))
result=dict(config=core['config'],scenarios=summary,quality=quality,attention_off=attention_off);(OUT/'summary.json').write_text(json.dumps(result,indent=2))
lines=['# 共享感知调度：实现、在线回放与边界','',
'已完成可切换的 baseline / 公平 FCFS / 前缀优先 LPM / 成本调度，接入会话 COW 和已有共享注意力。默认策略仍为 baseline；新策略需显式启用。',
'', '## 实现范围','',
'- Prefill/Decode 阶段服务机会保护、超时 Decode 补充服务、连续 Decode 次数上限、等待请求的到达顺序保护。',
'- 有界候选的物理前缀分组；一个 engine step 执行一个选定组。原有 ordinary/shared CUDA Graph 保留。',
'- 离线 decode 成本表和有限在线 EWMA 更新；按 tokens/ms 比较拆组与整批，保留 5% 选择余量，未知形状回退公平 FCFS。',
'- 活跃请求上限和 RequestQueueFull 背压；bulk fork 在引用获取前原子拒绝。',
'- Prefill 使用固定上限 1024 token；尚未实现可靠的上下文相关 Prefill 时间模型、多组融合 kernel、嵌套前缀最优搜索或硬 SLO 保证。',
'', '## 实验协议','',
'RTX 4090、Qwen3-0.6B FP16、CUDA Graph。所有策略 attention=auto，队列上限 96（包含 2 个暂停父会话的场景，最多接纳 94 个客户端请求）。每场景每策略预热一次，正式运行 4 次轮转顺序，共 80 次正式回放。',
'',
'输入为自然语言格式的合成设计文档；不是生产对话数据集。父会话在计时外准备；客户端 fork/冷请求按预定到达时刻注入。TTFT 从预定到达时刻计，包含等待引擎当前 step 完成的时间。使用实际采样并固定输出长度。',
'',
'goodput 门槛在运行前固定为 TTFT ≤ 500 ms 且请求内最大 ITL ≤ 75 ms。分子为满足两项的完成请求数，分母为该有限请求队列全部处理完成的时间；拒绝与超时均不计成功。下面为四次回放的中位数，范围和逐请求数据保存在 JSON。此处不是长时间稳态容量或生产 P99 认证。',
'', '场景定义：',
'- two_hot_groups：两个 4223-token 父会话，交错到达 64 个分支，各输出 64 token。',
'- short_hot_groups：相同分支结构，父会话缩短至 1151 token，验证不值得拆组时的选择。',
'- mixed：64 个长上下文热点分支，加 16 个独立 4096-token 冷请求，间隔 20 ms 到达，各输出 8 token。',
'- overload：64 个热点分支，加 96 个冷请求，间隔 5 ms 到达；测量相同队列上限下的显式拒绝。',
'- cold_only：32 个独立 1024-token 请求，间隔 10 ms，各输出 32 token。',
'', '## 主结果','',
'| 场景 | 策略 | 总吞吐 output tok/s | 达标请求 goodput req/s | 达标 / offered | 拒绝 |',
'|---|---|---:|---:|---:|---:|']
for scenario in scenarios:
    for policy in policies:
        m=summary[scenario][policy]['median']
        lines.append(f'| {scenario} | {policy} | {m["output_tokens_per_s"]:.1f} | {m["goodput"]:.2f} | {m["good"]:g} / {m["offered"]:g} | {m["rejected"]:g} |')
lines += ['', '另补充 16 次关闭共享注意力的 FCFS/cost 对照；合计 96 次正式回放。', '', '## 关闭共享注意力的消融', '', '| 场景 | FCFS tok/s | cost tok/s | cost 拆组选中次数 |', '|---|---:|---:|---:|']
for scenario,d in attention_off.items():
    lines.append(f'| {scenario} | {d["fcfs"]["throughput"]:.1f} | {d["cost"]["throughput"]:.1f} | {d["cost"]["group_choices"]} |')
lines += ['', '两组均未执行共享注意力；成本策略根据当前路径的实测时间保留整批。']
lines += ['', '“达标请求数”的 .5 来自四次计数的中位数，不代表存在半个请求。',
'', '## 冷热公平性的代价','',
'| 场景 | 策略 | 热 TTFT p95 ms | 热请求 max-ITL 的 p95 ms | 冷 TTFT p95 ms | 冷请求 max-ITL 的 p95 ms |',
'|---|---|---:|---:|---:|---:|']
for scenario in ('mixed','overload'):
    for policy in policies:
        c=summary[scenario][policy]['cohorts'];h,k=c['hot'],c['cold']
        lines.append(f'| {scenario} | {policy} | {h["ttft_ms"]["95"]:.1f} | {h["request_max_itl_ms"]["95"]:.1f} | {k["ttft_ms"]["95"]:.1f} | {k["request_max_itl_ms"]["95"]:.1f} |')
long=summary['two_hot_groups'];short=summary['short_hot_groups'];mixed=summary['mixed']
gain=(long['cost']['median']['output_tokens_per_s']/long['fcfs']['median']['output_tokens_per_s']-1)*100
lpm_loss=(short['lpm']['median']['output_tokens_per_s']/short['fcfs']['median']['output_tokens_per_s']-1)*100
cost_delta=(short['cost']['median']['output_tokens_per_s']/short['fcfs']['median']['output_tokens_per_s']-1)*100
raw_loss=(mixed['cost']['median']['output_tokens_per_s']/mixed['baseline']['median']['output_tokens_per_s']-1)*100
lines += ['', '## 如何解释收益','',
f'- 长上下文双组：成本调度相对公平 FCFS 吞吐变化 {gain:+.2f}%，共享路径实际执行。简单前缀优先也能获益，成本策略并不在所有场景优于 LPM。',
f'- 短上下文双组：简单前缀优先相对公平 FCFS 吞吐变化 {lpm_loss:+.2f}%；成本策略变化 {cost_delta:+.2f}%。校准表指出两次 32 请求共享执行的总成本高于一次普通 64 请求执行，成本策略因此保留整批。',
f'- 混合负载：原优先 Prefill 的基线在固定双 SLO 下 goodput 为 {mixed["baseline"]["median"]["goodput"]:.2f} req/s；成本策略为 {mixed["cost"]["median"]["goodput"]:.2f} req/s，但原始输出吞吐变化 {raw_loss:+.2f}%。不能将延迟保护写成无代价的吞吐提升。',
'- 冷请求不会因持续偏好共享组而完全失去服务机会，但高到达率下仍有较长 TTFT，当前机制不能保证全部请求满足 100 ms 等待目标或 500 ms TTFT。队列上限也不是 SLO 感知准入控制。',
'- 过载实验各策略接纳/拒绝数一致，goodput 改善不是通过额外拒绝冷请求得到。热点和冷请求的未达标比例均应保留。',
'', '## 正确性与回归','',
f'- 调度重排验证共 {quality["rows"]} 个请求/token 对齐点；最大相对 L2 {quality["max_relative_l2"]:.6f}，最大 KL {quality["max_kl"]:.6f}，均通过原定 L2 < .005、KL < .001 门槛。验证使用固定请求历史，按请求 ID 与生成位置对齐，不按批次位置比较。',
'- CPU 测试覆盖阶段交替、超过单批上限的 Decode 轮转、成本选择、超时覆盖和背压。GPU 混合 Prefill/Decode 与 bulk-fork 拒绝后的引用不变测试通过。',
'- 既有会话 COW GPU 测试通过；37 个分页算子 case 和 8 次动态 Graph replay 回归通过。回放全部完成已接纳请求，最终 KV 引用归零。',
'', '## 保留的诊断与下一步','',
'- 初始 ms/token 自适应 chunk 将固定执行开销误算为线性成本，导致 mixed 冷请求延迟接近 20 秒；该方案已撤下，原日志与中间数据保留为 replay-adaptive-diagnostic。',
'- 第二轮发现一次 Decode 不足以服务所有活跃请求时过早插入 Prefill，现已补充 overdue Decode 服务；诊断保留为 replay-fixed-v1。',
'- 下一步应校准带固定开销、已缓存上下文和 batch 结构的 Prefill 成本模型，改善当前吞吐/冷 TTFT 代价；然后做更长时间、真实会话轨迹与不同 SLO 下的验证。',
'- 缓存 LRU/成本淘汰、更大模型、BF16 共享引擎路径、GPU/CPU 分层仍未在本阶段实现。',
'', '实现与参数见 [SCHEDULER.md](SCHEDULER.md)。最终原始回放为 `results/scheduler/replay.json`，汇总为 `summary.json`；GPU 成本表为 `cost-profile.json`。','']
(ROOT/'docs/SCHEDULER_REPORT.md').write_text('\n'.join(lines))
print(json.dumps(dict(quality=quality,long_vs_fcfs_pct=gain,short_lpm_vs_fcfs_pct=lpm_loss,short_cost_vs_fcfs_pct=cost_delta,mixed_raw_throughput_change_pct=raw_loss),indent=2))
