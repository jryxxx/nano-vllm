"""Generate the release report from completed, versioned measurements."""
import json,statistics,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];data=root/'results/finalization'
def read(n):return json.loads((data/(n+'.json')).read_text())
s=read('summary')
assert len(s['sessions'])==22 and all(r['n']==3 for r in s['sessions'])
assert len(s['openloop'])==25 and all(r['n']==3 for r in s['openloop'])
assert len(s['combined'])==42 and all(r['n']==3 for r in s['combined'])
assert len(s['external'])==10 and all(r['n']==3 for r in s['external'])
for name in ['core','sessions','cache','tier']:
 assert all(r['status']=='passed' for r in read('suite-'+name))
for name in ['prefill-gpu','tier-validation','cow-gpu','combined-validation','scheduler-final','norm-bitwise']:
 assert read(name)['passed']
assert read('cache-logits')['full_recompute_gate_passed']
assert read('greedy')['exact_sequence_matches']==33
hashes={r['code_sha256'] for name in ['core','sessions','cache','tier'] for r in read('suite-'+name)}
h=hashlib.sha256()
for p in sorted((root/'nanovllm').rglob('*.py')):h.update(p.read_bytes())
assert hashes=={h.hexdigest()},'measured source drift'
def table(rows,cols):
 out=['| '+' | '.join(label for key,label in cols)+' |','|'+'|'.join('---' for _ in cols)+'|']
 for r in rows:
  vals=[r.get(k,'—') for k,label in cols];out.append('| '+' | '.join(f'{v:.2f}' if isinstance(v,float) else str(v) for v in vals)+' |')
 return '\n'.join(out)
def row(group,**kw):return next(r for r in s[group] if all(r[k]==v for k,v in kw.items()))
long=[row('combined',scenario='long',variant=v) for v in ['copy_off','cow_off','cow_auto']]
mix=[row('openloop',scenario='mixed',variant=v) for v in ['baseline','fixed_v1','fixed_large','model_only','joint']]
cache=read('cache-logits');gpu=read('prefill-gpu');cachechecks=[r for c in cache['cases'] for r in c['checks']];gpuchecks=[r for c in gpu['cases'] for r in c['checks']]
audit=dict(source_sha256=h.hexdigest(),cache_logit_checks=len(cachechecks),cache_max_relative_l2=max(r['relative_l2'] for r in cachechecks),prefill_logit_checks=len(gpuchecks),prefill_max_relative_l2=max(r['relative_l2'] for r in gpuchecks),norm_bitwise_cases=read('norm-bitwise')['bitwise_cases'],tier_bitwise_checks=read('tier-validation')['bitwise_checks'],paged_cases=len(read('paged-final')['cases']),graph_replays=len(read('paged-final')['graph_replay']),combined_cases=len(read('combined-validation')['cases']),greedy_tokens=read('greedy')['total_tokens'])
(data/'release-validation.json').write_text(json.dumps(audit,indent=2))
text=f'''# 共享上下文推理引擎：最终实现与实验报告

日期：2026-11-15。基于原始 nano-vLLM 的单 GPU 系统工程项目。主线 P0–P4、进阶 CPU 分层 P5、资源允许的 4B/FlashInfer/vLLM 参考及复现交付均已完成。硬件计数器限制仍存在，未把理论 KV 加载减少写成实测 HBM/L2 收益。

本报告使用最终精度配置 `fp32-residual-graph-fusion-fullaccum-v4`。历史阶段及失败候选数据保留作证据，不混入最终性能统计。原始数据在 [results/finalization](../results/finalization/)，聚合表为 [summary.json](../results/finalization/summary.json)，所有图均同时提供 PNG/SVG。推理源码 SHA256（按路径排序拼接 Python 内容）：`{h.hexdigest()}`。

## 1. 实现结果与主要结论

实现了通用分页/变长共享前缀 Decode、FP32 LSE 合并、自动回退与 Graph 捕获；实现了显式会话分支 COW、精确链式缓存身份、引用账本、取消回收和多种淘汰策略；实现了 Prefill/Decode 成本模型、延迟保护、有界共享分组和过载准入；增加了有界 CPU pinned 缓存、传输状态机、恢复/重算选择和故障回滚。

- 32 分支、4223-token 父历史、每分支 64 输出的受控实验，包含冷父请求 Prefill：全拷贝+普通 Attention 为 {long[0]['cold_tokens_per_s']:.1f} tok/s，COW+普通为 {long[1]['cold_tokens_per_s']:.1f}，COW+auto 为 {long[2]['cold_tokens_per_s']:.1f}。组合相对全拷贝为 {long[2]['cold_tokens_per_s']/long[0]['cold_tokens_per_s']:.2f}×，共享 Attention 在相同 COW 上额外提升 {(long[2]['cold_tokens_per_s']/long[1]['cold_tokens_per_s']-1)*100:.1f}%。这是相对显式全拷贝控制组，不是相对上游的加速比。
- 同一分支实验的峰值活跃物理 KV 页从 {long[0]['peak_live_pages']} 降为 {long[2]['peak_live_pages']}（减少 {(1-long[2]['peak_live_pages']/long[0]['peak_live_pages'])*100:.1f}%）。预分配 KV 池相同；不能将活跃页下降写成进程显存下降。
- 混合开放到达负载，联合调度相对早期固定 1024 chunk 策略，吞吐提升 {(mix[-1]['output_tokens_per_s']/mix[1]['output_tokens_per_s']-1)*100:.1f}%，冷请求 P95 TTFT 降低 {(1-mix[-1]['cold_ttft_p95_ms']/mix[1]['cold_ttft_p95_ms'])*100:.1f}%。相对最终版 baseline，联合策略的总吞吐仍较低，但固定 SLO goodput 更高。
- 不存在全负载通吃结论：因果多轮负载的 Full 仍慢于冻结上游；CPU 强制恢复在当前 0.6B/PCIe 配置没有吞吐收益；FlashInfer 在全部本轮微基准形状上快于自研共享路径。默认不强制启用优化。

## 2. 架构与资源所有权

```mermaid
flowchart TD
    A[请求 / 会话 / 分支 / 取消] --> B[Engine]
    B --> C[Block Manager: 精确前缀身份与所有权]
    C --> D[Scheduler: 到期保护 / 成本 / 有界候选]
    D --> E[Runner: 页表 / 重排 / Graph]
    E --> F{{Attention Dispatcher}}
    F --> G[普通 FlashAttention]
    F --> H[共享前缀 + 私有后缀 + FP32 LSE 合并]
    C <--> T[Transfer Manager]
    T <--> P[有界 CPU pinned pool]
```

逻辑前缀节点由父节点 UID 和精确 token 元组弱驻留，hash 只作索引；GPU 物理槽有独立 generation 和位置关联。每个请求、保存会话和传输票据拥有显式引用。只发布已经写完的 KV，不把最后采样、尚未参与计算的 token 计作有效 KV。完整共享页只读；共享尾页在写前复制已计算部分，失败回滚分配与引用。

共享 Attention 按当前批次共同的完整物理页划分前缀和后缀。每个请求保留自己的 Q/softmax；两个片段输出以 FP32 log-sum-exp 合并。分页可以不连续，后缀变长；空段、Graph padding、请求重排与路径切换有独立测试。

调度器用有界候选查询和前缀分组，先处理服务间隔和等待保护，再按成本选择批次；不等待未来请求扩组。Prefill 模型同时考虑 query、cached context、batch、attention 工作量，采用测量表、受限局部残差修正和在线平滑。Decode 是形状邻域查表。缺少可用预测时回退有界 FCFS。首版采用最小共享前缀分桶和桶内最长公共前缀，并不求解任意嵌套前缀的全局最优分组。

CPU 分层只接受零活跃引用、不可变完整页。主流先复制到 GPU 暂存，传输流等待事件后 D2H；恢复预留 GPU 目标并持有引用，H2D 完成后才发布索引。COPYING 状态不能被淘汰。取消、复制失败与分配失败均执行回滚。保存会话仍常驻 GPU；当前不迁移活跃会话。

## 3. 数值基础与正确性验收

原先 compiled/eager RMSNorm、FP16 残差和低精度 GEMM 归约组合在跨 Prefill 形状比较中超过既定门槛。本轮没有放宽门槛，而是采用 FP32 残差、禁用 FP16 reduced-precision GEMM reduction；Graph 只融合 RMSNorm 的逐元素阶段，保留 PyTorch mean/rsqrt。Prefill 仍用稳定 eager 实现。独立 Hugging Face FP32（禁用 TF32）参考的 6 个样本，平均 logits 相对 L2 从约 0.00169 降至 0.00126；这是数值证据，不是性能提升归因。

| 验收 | 最终结果 |
|---|---|
| 缓存路径与完整重算 logits | {len(cachechecks)} 次，最大相对 L2 {audit['cache_max_relative_l2']:.6f} |
| 五种 Prefill/调度路径 | {len(gpuchecks)} 次，最大相对 L2 {audit['prefill_max_relative_l2']:.6f} |
| 共享 Attention 0.6B / 4B | 同轨迹 L2 < 0.005、KL < 0.001；0.6B 自动与 4B force 均通过 |
| RMSNorm Graph/eager 等价 | {audit['norm_bitwise_cases']} 个逐位检查，含 FP16/BF16、FP32 残差、strided 输入 |
| 分页算子 | {audit['paged_cases']} 个边界用例、{audit['graph_replays']} 次动态 Graph replay |
| COW 与联合路径 | 18 个整模型组合；逐位复制、源快照不变、尾页边界、失败回滚、压力恢复通过 |
| GPU 驻留缓存 | FIFO/LRU/leaf-LRU/cost 共 9267 个逐位检查 |
| CPU 分层 | {audit['tier_bitwise_checks']} 个逐位检查；真实拷贝后异常、传输中取消、引用清理通过 |
| 贪心回归 | 33 条、528 token 全部一致；测试专用 argmax sampler |
| 调度与身份 | 强制 hash 冲突、6000 次缓存状态操作、请求重排、队列拒绝回滚通过 |

严谨的数值门槛建立在相同 teacher-forced 历史上；自由生成回归是补充。上述覆盖不等于任意模型/任意长度的普遍证明。BF16 当前共享专用路径回退普通 Attention，不能把回退的零误差说成 BF16 共享 kernel 集成已启用。

Prefill 8 个 holdout 的实际预测器 MAPE 为 {read('prefill-model')['mape']*100:.2f}%；独立新增 6 个形状未参与拟合，MAPE {read('fresh-audit')['mape']*100:.2f}%，最大 {read('fresh-audit')['max_relative_error']*100:.2f}%。成本表绑定精度版本、模型 config SHA、Torch/CUDA/GPU 和执行模式，失配直接拒绝，需重标定。

## 4. 测量协议与对照边界

RTX 4090 24GB、同一主模型/权重、FP16 KV、同一最终精度实现；精确环境见 [锁文件及模型校验](../results/finalization/environment/)。每配置完整 trace 预热排除后，至少 3 次测量，表格为三次中位数，CSV/图中另含 min–max。只 3 次重复，不作显著性或 P99 普遍保证。

B0 是冻结原始上游，仅加 192 页 KV 容量上限和观测包装；B1 是公共正确性/生命周期/埋点底座、FCFS 阶段保护、FIFO、普通 Attention；B2 为 LPM。M=B1+会话 COW+LRU，K=B1+auto，S=成本调度/普通 Attention。Full=COW+LRU+auto+成本调度；Tiered 再加 CPU。Full-no-M/no-K/no-S 是对应逐项关闭。B0→B1 同时涉及精度与调度/管理，不能把差异全归为共享算子。

因果会话回放为 32 会话×3 轮、每轮 16 输出；下一轮仅在上一轮完成+20ms 工具等待后产生，使用真实格式文本和固定 teacher-forced 响应。到达时刻因此随策略内生产生，不能伪称各策略同一开环时间戳。单独的开放回放固定外生到达，并包含混合/过载/冷请求/两热点/短热点。TTFT 从计划到达时刻计时；goodput 同时要求 TTFT≤500ms、请求内最大 ITL≤75ms，队列上限 96。拒绝和未完成单独统计。单请求 TPOT 从原始 token 时间计算，见 CSV，不与最大 ITL 混用。

静态外部参考采用 32 请求、1024 前缀+64 私有 token、64 输出；共享前缀 seed 计时前写入，故它是 warm-prefix static-batch 参考。no_shared 不 seed。vLLM 0.8.5 V0 使用 32-token 页、Torch 2.6/CUDA12.4，nano 使用 256-token 页、Torch2.5.1/CUDA12.1，KV token 容量均为 49152；实际配置在 JSON。不同后端/版本/通用服务开销使该对比不能用于单特性归因。

## 5. COW 与共享 Attention 组合

冷父 Prefill、fork、分支生成和清理全部计入 cold throughput。全拷贝控制有意取消共享物理页，从而也不具备共享 Attention 条件；`eager_tail` 控制只把相同尾页复制提前到 fork，保留完整共享页，进一步分离 COW 时机与共享算子作用。

{table([r for r in s['combined'] if r['variant'] in ['copy_off','cow_off','cow_auto']], [('scenario','场景'),('variant','配置'),('cold_tokens_per_s','冷端到端 tok/s'),('peak_live_pages','峰值活跃页'),('copy_bytes','复制 bytes')])}

![分支消融](../results/finalization/figures/branch-ablation.png)

## 6. 因果多轮整体消融

{table(s['sessions'], [('scenario','场景'),('variant','配置'),('output_tokens_per_s','输出 tok/s'),('prefill_tokens','实际 Prefill token'),('ttft_p95_ms','TTFT P95 ms'),('goodput','goodput req/s')])}

![会话消融](../results/finalization/figures/session-ablation.png)

Full 在这两种会话 trace 都没有超过 B0；shared 场景 Full-no-S 优于 Full，说明更复杂调度的额外开销和分批策略可以抵消收益。M 在 shared 场景把实际 Prefill 从 B1 的 5314 降到 2550，但不能推广为所有调度下同比例节省。Full shared 的 route 计数没有触发共享 kernel；K/Full 名称代表允许使用，不代表每个批次实际启用。具体启用次数见每次试验 `route_counts`。

## 7. Prefill 与在线调度

{table(s['openloop'], [('scenario','负载'),('variant','配置'),('output_tokens_per_s','输出 tok/s'),('goodput','goodput req/s'),('cold_ttft_p95_ms','冷 TTFT P95 ms'),('rejected','拒绝数'),('unfinished','未完成数')])}

![开放回放 goodput](../results/finalization/figures/openloop-goodput.png)

联合策略相对 fixed_v1 改善冷请求等待和吞吐；baseline 在部分场景拥有更高原始吞吐/更低冷 TTFT，但长 Prefill 会打断在线 Decode，从而未必满足请求级 ITL 标准。过载各策略仍有拒绝及 SLO 违反，不声称绝对等待上界。短热点负对照 joint 吞吐约比 baseline 低 2.9%，本次低于早期 5% 目标，但因果多轮整体不是此结论的适用范围。

## 8. 缓存策略与 CPU 分层

FIFO/LRU/leaf-LRU/成本淘汰在 20/32/64 页、污染/热点切换/流式/混合成本四种 trace 上测试，共 144 个试验。策略结果见 [cache-summary.csv](../results/finalization/cache-summary.csv)。没有证据支持一种淘汰策略在所有场景稳定胜出，因此默认 FIFO，Full 显式使用 LRU。

{table(s['tier'], [('scenario','负载'),('mode','恢复模式'),('output_tokens_per_s','输出 tok/s'),('restored_pages','恢复页'),('unique_cached_logical_pages','唯一缓存逻辑页'),('pinned_bytes','pinned bytes'),('staging_bytes','额外 GPU 暂存 bytes')])}

GPU 池固定 20 页（560 MiB），CPU 64 页（1.75 GiB），此性能实验的 16 暂存页另占 448 MiB GPU；不是相同总 GPU 占用比较。按实际非连续 KV 目标布局标定，单页 H2D 约 1.22ms，D2H 约 1.13ms。auto 在这些 trace 均选择重算，没有 H2D 恢复；always 完成恢复但吞吐更低。缓存容量与传输能力完成验收，不宣称 0.6B 分层吞吐加速。

受控 profiler 实验记录 {read('transfer-overlap')['dtoh_events']} 个 D2H 事件，与 Decode kernel 的区间并集重叠 {read('transfer-overlap')['overlap_us']:.1f}µs，传输 {read('transfer-overlap')['transfer']['d2h_bytes']} bytes。见 [时间线](../results/finalization/transfer-trace.json) 与 [重叠摘要](../results/finalization/transfer-overlap.json)。这是后台 spill 的重叠证据，不是完整 serving 吞吐证据。无 GPU 硬件 counters/nsys，因此不报告 HBM/L2/occupancy 实测。

## 9. 算子与外部参考

FlashInfer 0.2.5 共 16 个形状×3 后端×eager/Graph；包括共享+后缀+merge 的完整 GPU 路径，固定元数据 planning 排除但另记。初次 JIT/warmup 不计时，三次各 100 次调用；前 4 个 query 对 FP32 参考验收。共享前缀为连续布局，该组不是通用分页全覆盖（分页由独立正确性测试负责）。

![完整 Attention 路径](../results/finalization/figures/attention-reference.png)

Q=16/KV=8/head_dim=128、B=64、prefix=4096、suffix=128、Graph：FlashAttention 490.22µs，自研共享 142.99µs（3.43×），FlashInfer 35.83µs。小批量自研路径会退化，auto 回退；FlashInfer 在本轮全部形状更快。共享分解与 LSE 合并不是本项目首次提出。

{table(s['external'], [('model','模型'),('scenario','场景'),('backend','引擎/路径'),('output_tokens_per_s','输出 tok/s'),('ttft_p95_ms','TTFT P95 ms'),('itl_p95_ms','ITL P95 ms')])}

![外部静态参考](../results/finalization/figures/external-reference.png)

0.6B shared nano-auto 相对 nano-off 吞吐约 +10.8%；4B force 约 +5.1%，无共享场景基本相同。4B 自动选择尚未作完整边界标定，仍默认回退，不能将少数 force 结果扩大为已全面自动支持。vLLM 数字仅代表本机固定 V0 配置的静态参考，不代表其最佳调优性能或在线服务排名。

## 10. 复现、范围与贡献定位

入口与会话 API 示例见 [README.shared-context.md](../README.shared-context.md)。`bash experiments/reproduce.sh verify` 执行正确性，`full` 串行重标定及全套实验，`summarize` 重建数据表/图。性能结果严格来自远程服务器，不使用本地开发机替代 GPU 测量。归档包含源代码、补丁、测试、原始 JSON/CSV、图、时间线、环境锁与逐文件 SHA256；权重/虚拟环境不在归档中。

本次范围已完成；以下是明确边界或后续研究方向：仅单 GPU/单线程 step 边界 API；BF16 专用集成未启用；4B 自动策略未全面标定；CPU 不迁移保存/活跃会话；成本模型与分组为有界启发式而非全局最优；未覆盖真实生产请求分布；不保证任意到达负载的 SLO；多卡、MoE、KV 量化、LoRA、推测解码、多层前缀树 kernel 为原方案排除项。性能负结果和硬件 counters 限制不应从展示中删去。

参考：原始 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)、[Hydragen](https://arxiv.org/abs/2402.05099)、[FlashInfer Cascade](https://docs.flashinfer.ai/api/cascade.html)、[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)、[vLLM 0.8.5 安装与二进制兼容说明](https://docs.vllm.ai/en/v0.8.5/getting_started/installation/gpu.html)。本项目定位是有独立实现与实验论证的系统工程，不宣称算法首创或 SOTA。
'''
(root/'docs/FINAL_REPORT.md').write_text(text)
print(json.dumps(audit,indent=2))
