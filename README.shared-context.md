> 新机器运行请先看 [仓库首页](README.md)；本文件保留原实验服务器目录和操作说明。

# Shared-context nano-vLLM

基于原始 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的单 GPU 系统工程项目：共享分页 Attention、会话分支 KV 生命周期、有等待约束的成本调度，以及可选 CPU 缓存。冻结上游提交 `bb823b3e06983d71485a8e1f23715ebd87d98ef8`；上游已有分页、前缀缓存和 CUDA Graph 不作为新增贡献。

完整结果见 [最终技术报告](docs/FINAL_REPORT.md)，验收状态见 [清单](docs/FINALIZATION_STATUS.md)。原始数据与可导出图表在 [results/finalization](results/finalization/)。历史阶段报告保留当时结论；最终精度配置为 `fp32-residual-graph-fusion-fullaccum-v4`。

## 运行与复现

记录服务器：RTX 4090 24GB，单 GPU，Linux，Python 3.10，PyTorch 2.5.1+cu121 / Triton 3.1.0 / FlashAttention 2.7.4.post1。精确依赖、驱动及模型文件 SHA256 见 [environment](results/finalization/environment/)。GPU 性能计数器未开放，实验使用 CUDA Events 和 PyTorch profiler。

```bash
ssh Nano-vLLM
cd /root/autodl-tmp/shared-context/src/nano-vllm-shared
# 正确性回归，使用已标定且带环境签名的成本表
bash experiments/reproduce.sh verify
# 全套串行实验：重标定、主线、逐项关闭、分支、4B、vLLM、FlashInfer、图表
bash experiments/reproduce.sh full
# 只根据已有原始数据重建 CSV、汇总 JSON、PNG/SVG
bash experiments/reproduce.sh summarize
```

`full` 会覆盖 `results/finalization` 中同名结果；已交付归档保留本轮原始数据。不要与其他 GPU 负载同时运行。各性能配置完整预热后至少独立重复 3 次，外部环境单独启动。脚本采用当前服务器目录布局；迁移时需修改脚本中的 `BASE` 和 Python/model 路径，并重新标定，不能只改旧 JSON 的签名。

主环境在 `../../envs/main`；vLLM 在 `../../envs/vllm-reference`；FlashInfer 通过独立目录 `PYTHONPATH` 加载。模型在 `../../models`。归档不含权重和虚拟环境，环境重建需先安装对应 CUDA/PyTorch 与 FlashAttention，再按锁文件核对，不能把参考引擎依赖直接装进主环境。vLLM 参考锁定 0.8.5 / Torch 2.6.0 / CUDA 12.4 / Transformers 4.51.3；FlashInfer 0.2.5 与主环境 Torch 配合，JIT 需要 nvcc。

## 显式启用功能

默认保留普通 Attention、baseline 调度、FIFO，CPU 缓存关闭。`auto` 仅在已标定的 RTX 4090、Qwen3-0.6B FP16、CUDA Graph、至少 32 请求、至少 1024 共同前缀 token、后缀不长于前缀时启用共享路径。条件不符回退普通 FlashAttention。4B `force` 已验证，`auto` 保持回退；BF16 当前使用普通路径。

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/root/autodl-tmp/shared-context/models/Qwen3-0.6B-fp16",
    max_num_seqs=64, max_model_len=4096,
    max_num_batched_tokens=8192, gpu_memory_utilization=0.65,
    shared_attention_backend="auto",
    schedule_policy="cost",
    schedule_prefill_mode="model", schedule_phase_mode="predictive",
    schedule_cost_profile="results/finalization/cost-profile.json",
    cache_policy="lru", schedule_queue_limit=96,
)
try:
    request = llm.add_request("Explain copy-on-write.",
        SamplingParams(max_tokens=32), session_id="conversation")
    while not llm.is_finished():
        llm.step()
    suffix = llm.tokenizer.encode("\nUser: Give an example.\nAssistant:")
    branch = llm.fork_session("conversation", SamplingParams(max_tokens=32),
        append_token_ids=suffix, output_session_id="branch")
    while not llm.is_finished():
        llm.step()
    llm.close_session("conversation")
    llm.close_session("branch")
finally:
    llm.exit()
```

API 在 step 边界调用，当前是单线程、单实例、单 GPU 使用约定。`fork_request(s)` 可分叉活跃请求，`pause_request/resume_request` 表达工具等待，`cancel_request` 释放活跃所有权。保存会话保留 GPU 引用，必须显式关闭；CPU 分层只迁移零引用完整缓存页，不能把保存会话理解成自动 CPU offload。标准采样 API 沿用上游正温度限制；贪心回归用测试专属 argmax sampler。

CPU 分层可选参数：`cpu_cache_pages=64, cpu_cache_staging=4, cpu_cache_restore="auto"`。0.6B FP16 每 256-token 页占 28 MiB：64 CPU 页是 1.75 GiB pinned RAM，4 暂存页额外占 112 MiB GPU。实际 GPU KV 页池与暂存独立计费，`max_kvcache_blocks` 可限制前者。强制恢复 `always` 只用于消融，不是推荐默认值。

## 代码入口

- `engine/block_manager.py`：精确链式身份、引用所有权、COW、FIFO/LRU/leaf-LRU/成本淘汰。
- `engine/shared_plan.py`、`layers/shared_prefix.py`：共同完整页规划、共享/后缀/LSE 合并。
- `engine/scheduler.py`、`engine/schedule_cost.py`：有界候选、阶段保护、Prefill 查表与局部修正、在线平滑、队列上限。
- `engine/transfer_manager.py`：pinned pool、GPU 暂存、传输事件、恢复预算及失败回滚。
- `layers/layernorm.py`：FP32 残差、保持 PyTorch reduction 的 Graph 专用融合。
- `experiments/run_release_measurements.sh`、`run_supplement.sh`：实验协议和原始命令；`summarize_release.py`：数据汇总与绘图。

不宣称普遍优于成熟引擎。FlashInfer 在本轮全部受测共享形状上更快；完整方案也有负收益场景。项目价值在实现、数值与资源安全、分层对照以及明确的启用边界。多 GPU、MoE、KV 量化、LoRA、推测解码、多层前缀树 kernel 不在本次范围。
