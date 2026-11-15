# 共享感知调度：实现、在线回放与边界

已完成可切换的 baseline / 公平 FCFS / 前缀优先 LPM / 成本调度，接入会话 COW 和已有共享注意力。默认策略仍为 baseline；新策略需显式启用。

## 实现范围

- Prefill/Decode 阶段服务机会保护、超时 Decode 补充服务、连续 Decode 次数上限、等待请求的到达顺序保护。
- 有界候选的物理前缀分组；一个 engine step 执行一个选定组。原有 ordinary/shared CUDA Graph 保留。
- 离线 decode 成本表和有限在线 EWMA 更新；按 tokens/ms 比较拆组与整批，保留 5% 选择余量，未知形状回退公平 FCFS。
- 活跃请求上限和 RequestQueueFull 背压；bulk fork 在引用获取前原子拒绝。
- Prefill 使用固定上限 1024 token；尚未实现可靠的上下文相关 Prefill 时间模型、多组融合 kernel、嵌套前缀最优搜索或硬 SLO 保证。

## 实验协议

RTX 4090、Qwen3-0.6B FP16、CUDA Graph。所有策略 attention=auto，队列上限 96（包含 2 个暂停父会话的场景，最多接纳 94 个客户端请求）。每场景每策略预热一次，正式运行 4 次轮转顺序，共 80 次正式回放。

输入为自然语言格式的合成设计文档；不是生产对话数据集。父会话在计时外准备；客户端 fork/冷请求按预定到达时刻注入。TTFT 从预定到达时刻计，包含等待引擎当前 step 完成的时间。使用实际采样并固定输出长度。

goodput 门槛在运行前固定为 TTFT ≤ 500 ms 且请求内最大 ITL ≤ 75 ms。分子为满足两项的完成请求数，分母为该有限请求队列全部处理完成的时间；拒绝与超时均不计成功。下面为四次回放的中位数，范围和逐请求数据保存在 JSON。此处不是长时间稳态容量或生产 P99 认证。

场景定义：
- two_hot_groups：两个 4223-token 父会话，交错到达 64 个分支，各输出 64 token。
- short_hot_groups：相同分支结构，父会话缩短至 1151 token，验证不值得拆组时的选择。
- mixed：64 个长上下文热点分支，加 16 个独立 4096-token 冷请求，间隔 20 ms 到达，各输出 8 token。
- overload：64 个热点分支，加 96 个冷请求，间隔 5 ms 到达；测量相同队列上限下的显式拒绝。
- cold_only：32 个独立 1024-token 请求，间隔 10 ms，各输出 32 token。

## 主结果

| 场景 | 策略 | 总吞吐 output tok/s | 达标请求 goodput req/s | 达标 / offered | 拒绝 |
|---|---|---:|---:|---:|---:|
| two_hot_groups | baseline | 3531.3 | 55.18 | 64 / 64 | 0 |
| two_hot_groups | fcfs | 3515.0 | 54.92 | 64 / 64 | 0 |
| two_hot_groups | lpm | 3837.9 | 59.97 | 64 / 64 | 0 |
| two_hot_groups | cost | 3771.5 | 58.93 | 64 / 64 | 0 |
| short_hot_groups | baseline | 8092.4 | 126.44 | 64 / 64 | 0 |
| short_hot_groups | fcfs | 8012.3 | 125.19 | 64 / 64 | 0 |
| short_hot_groups | lpm | 6292.4 | 98.32 | 64 / 64 | 0 |
| short_hot_groups | cost | 7677.1 | 119.95 | 64 / 64 | 0 |
| mixed | baseline | 2173.8 | 0.00 | 0 / 80 | 0 |
| mixed | fcfs | 1168.7 | 18.26 | 66 / 80 | 0 |
| mixed | lpm | 1168.4 | 17.95 | 64.5 / 80 | 0 |
| mixed | cost | 1164.3 | 17.99 | 65.5 / 80 | 0 |
| overload | baseline | 1659.6 | 0.00 | 0 / 160 | 66 |
| overload | fcfs | 735.5 | 10.94 | 64.5 / 160 | 66 |
| overload | lpm | 737.7 | 11.06 | 65 / 160 | 66 |
| overload | cost | 760.1 | 11.45 | 65 / 160 | 66 |
| cold_only | baseline | 1708.4 | 7.51 | 4.5 / 32 | 0 |
| cold_only | fcfs | 695.2 | 10.85 | 16 / 32 | 0 |
| cold_only | lpm | 691.9 | 10.81 | 16 / 32 | 0 |
| cold_only | cost | 694.7 | 10.52 | 15.5 / 32 | 0 |

另补充 16 次关闭共享注意力的 FCFS/cost 对照；合计 96 次正式回放。

## 关闭共享注意力的消融

| 场景 | FCFS tok/s | cost tok/s | cost 拆组选中次数 |
|---|---:|---:|---:|
| two_hot_groups | 3526.3 | 3464.7 | 0 |
| short_hot_groups | 8013.8 | 7771.0 | 0 |

两组均未执行共享注意力；成本策略根据当前路径的实测时间保留整批。

“达标请求数”的 .5 来自四次计数的中位数，不代表存在半个请求。

## 冷热公平性的代价

| 场景 | 策略 | 热 TTFT p95 ms | 热请求 max-ITL 的 p95 ms | 冷 TTFT p95 ms | 冷请求 max-ITL 的 p95 ms |
|---|---|---:|---:|---:|---:|
| mixed | baseline | 762.5 | 18.0 | 440.0 | 1759.9 |
| mixed | fcfs | 54.7 | 73.0 | 3188.6 | 72.2 |
| mixed | lpm | 54.6 | 72.7 | 3185.7 | 72.2 |
| mixed | cost | 55.1 | 72.8 | 3197.7 | 72.4 |
| overload | baseline | 1384.3 | 18.0 | 1178.7 | 2346.8 |
| overload | fcfs | 55.0 | 73.1 | 5505.5 | 71.8 |
| overload | lpm | 60.4 | 72.5 | 5487.6 | 71.4 |
| overload | cost | 57.7 | 72.5 | 5321.5 | 71.5 |

## 如何解释收益

- 长上下文双组：成本调度相对公平 FCFS 吞吐变化 +7.30%，共享路径实际执行。简单前缀优先也能获益，成本策略并不在所有场景优于 LPM。
- 短上下文双组：简单前缀优先相对公平 FCFS 吞吐变化 -21.47%；成本策略变化 -4.18%。校准表指出两次 32 请求共享执行的总成本高于一次普通 64 请求执行，成本策略因此保留整批。
- 混合负载：原优先 Prefill 的基线在固定双 SLO 下 goodput 为 0.00 req/s；成本策略为 17.99 req/s，但原始输出吞吐变化 -46.44%。不能将延迟保护写成无代价的吞吐提升。
- 冷请求不会因持续偏好共享组而完全失去服务机会，但高到达率下仍有较长 TTFT，当前机制不能保证全部请求满足 100 ms 等待目标或 500 ms TTFT。队列上限也不是 SLO 感知准入控制。
- 过载实验各策略接纳/拒绝数一致，goodput 改善不是通过额外拒绝冷请求得到。热点和冷请求的未达标比例均应保留。

## 正确性与回归

- 调度重排验证共 1024 个请求/token 对齐点；最大相对 L2 0.004100，最大 KL 0.000027，均通过原定 L2 < .005、KL < .001 门槛。验证使用固定请求历史，按请求 ID 与生成位置对齐，不按批次位置比较。
- CPU 测试覆盖阶段交替、超过单批上限的 Decode 轮转、成本选择、超时覆盖和背压。GPU 混合 Prefill/Decode 与 bulk-fork 拒绝后的引用不变测试通过。
- 既有会话 COW GPU 测试通过；37 个分页算子 case 和 8 次动态 Graph replay 回归通过。回放全部完成已接纳请求，最终 KV 引用归零。

## 保留的诊断与下一步

- 初始 ms/token 自适应 chunk 将固定执行开销误算为线性成本，导致 mixed 冷请求延迟接近 20 秒；该方案已撤下，原日志与中间数据保留为 replay-adaptive-diagnostic。
- 第二轮发现一次 Decode 不足以服务所有活跃请求时过早插入 Prefill，现已补充 overdue Decode 服务；诊断保留为 replay-fixed-v1。
- 下一步应校准带固定开销、已缓存上下文和 batch 结构的 Prefill 成本模型，改善当前吞吐/冷 TTFT 代价；然后做更长时间、真实会话轨迹与不同 SLO 下的验证。
- 缓存 LRU/成本淘汰、更大模型、BF16 共享引擎路径、GPU/CPU 分层仍未在本阶段实现。

实现与参数见 [SCHEDULER.md](SCHEDULER.md)。最终原始回放为 `results/scheduler/replay.json`，汇总为 `summary.json`；GPU 成本表为 `cost-profile.json`。
