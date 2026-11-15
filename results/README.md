# 实验数据说明

## 版本与来源

`finalization/` 是 2026-11-15 最终实验，精度版本 `fp32-residual-graph-fusion-fullaccum-v4`。`release-validation.json` 记录实测引擎源码 SHA256；GitHub 发布时推理模块保持逐字节一致，仅增加可移植实验入口、历史回归 fixture 与文档。其他 results 目录和历史报告用于追溯此前阶段，不能与最终数字混为同一配置。

数据由项目脚本生成并在 RTX 4090 24GB 远程服务器测量。随机 token 用于算子 shape 压测；端到端请求采用程序生成的共享技术文档、私有问题与固定 token 响应。多轮回放遵守上一轮完成+20ms 工具等待依赖。无真实用户对话或私有业务数据；模型权重、虚拟环境、简历与凭据不在仓库。

## 文件索引

| 文件 | 内容 |
|---|---|
| `finalization/summary.json`、`*-summary.csv` | 分组中位数、min/max、重复次数，供表格/图重建 |
| `sessions-*.json` | B0/B1/B2/M/K/S/Full/Tiered 和 Full-no-M/K/S；每配置 2 场景×3 次，含原始 token 时间 |
| `combined-final.json` | 7 场景×6 配置×3 次；冷 Prefill、fork、完整生成和回收计时 |
| `openloop-final.json` | 5 策略×5 负载×3 次；热/冷 TTFT、请求最大 ITL、拒绝/未完成及 goodput |
| `cache-final.json` | 4 淘汰策略×3 GPU 页容量×4 负载×3 次，共 144 次 |
| `tier-off/auto/always.json` | 20 GPU 页、64 CPU 页、16 GPU 暂存页；传输与重算对照，额外内存独立记录 |
| `flashinfer-reference.json` | 16 形状×3 后端×eager/Graph；含 3 次计时样本及独立精度检查 |
| `reference-*.json` | 0.6B nano-off/auto、4B off/force、vLLM 0.8.5 静态参考 |
| `*-profile.json`、`fresh-audit.json` | 模型/运行时签名、标定数据、未参与拟合的新形状审计 |
| `transfer-trace.json` | PyTorch profiler 原始时间线，可在 Perfetto/Chrome trace viewer 中打开 |
| `figures/` | PNG/SVG，绘图入口 `experiments/summarize_release.py` |
| `environment/` | 主环境/参考环境锁文件、GPU/驱动/CUDA、模型文件 SHA256；本地 wheel 路径仅供追溯 |
| `patches/` | 从上游及阶段快照恢复**原实测快照**的补丁；不包含后续 GitHub 入口适配 |
| `norm-*`、`fused-*`、`hybrid-*`、`precision-*` | 数值诊断与失败候选，不能视作最终已启用优化 |

## 统计口径

完整 trace 预热不进入测量，性能配置各独立重复至少 3 次。表格取中位数，min/max 展示波动，不宣称统计显著性。TTFT 从计划到达计，包含排队；goodput 同时要求 TTFT≤500ms 和请求最大 ITL≤75ms。因果会话到达随上一轮完成而变化，固定开放到达实验单独统计。活跃 KV 页、预分配池和 GPU 暂存是不同资源指标。

FlashInfer 对照中固定 metadata planning 排除但另记，端到端分支计时包含 CPU 规划。vLLM 使用不同 Torch/CUDA/页大小，仅作固定配置参考。负结果：Full 因果会话不及冻结上游；CPU 强制恢复无吞吐收益；FlashInfer 在本轮算子形状更快。

## 完整性与重跑

`PUBLIC_DATA_SHA256.json` 对本次发布结果逐文件记录 SHA256；发布原始归档在 GitHub Release。重新运行会覆盖同名结果，因此需要保留原 Git commit，且不能将部分重跑数据与历史未重跑数据合并宣称一次完整新实验。GPU 硬件计数器不可用，时间线不能替代 HBM/L2 测量。
