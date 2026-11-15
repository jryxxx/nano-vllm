# 分页共享 Attention 与引擎集成

日期：2026-11-13。基于官方 nano-vLLM `bb823b3e06983d71485a8e1f23715ebd87d98ef8`。

远程原版：`/root/autodl-tmp/shared-context/src/nano-vllm`，保持未修改。

远程开发版：`/root/autodl-tmp/shared-context/src/nano-vllm-shared`，分支 `shared-prefix`。

## 已实现

- 共享前缀通过 block table 直接读取，不要求物理页连续，不先 gather KV。
- 私有后缀通过独立的分页表与每请求长度交给 FlashAttention，支持不同后缀长度、非整页尾部。
- FP32 LSE 合并处理空分段、Graph 填充行与零有效长度。
- 独立普通/共享 CUDA Graph；前缀长度、完整页表、后缀页表和有效长度在回放前更新。
- 单批次共同前缀识别、路由原因计数与元数据准备耗时记录。
- 三种模式：`off`、`auto`、`force`；默认 `off`。Prefill 和调度顺序保持原版行为。

## 自动选择规则

当前 `auto` 是按实测结果配置的保守规则，不是在线学习的成本模型。以下条件全部满足才启用：

1. 单 GPU、FP16、RTX 4090，16 Query Heads / 8 KV Heads / head_dim 128。
2. 使用 CUDA Graph，实际批次不超过其支持上限512。
3. 至少32个请求，全部共享至少1024 tokens的完整前缀块。
4. 最长私有后缀不超过共享前缀长度。

不满足时自动使用原始 Attention。双前缀组混在同一批且没有全批共同前缀时，整体回退；本轮没有实现组内重排或多组共享 kernel。

`force` 用于实验，可绕过执行模式、批次和前缀长度的收益规则，但仍要求受支持的单卡 FP16/head_dim，以及至少一个共同完整块。它不会绕过 BF16 精度门槛。

BF16 原始分段实现的整模型 logits 相对 L2 误差约1.78%，未通过预设0.5%门槛，因此整模型共享路径暂不对 BF16 开放。单算子 BF16 验证通过不能替代整模型验证。自动 BF16 回退的整模型输出已验证与原路径一致。

## FP16 的数值界限

固定 teacher-forced 轨迹中，Graph 自动模式覆盖批次33 → 22 → 11及共享/普通图切换。最大相对 L2 约0.264%，最大绝对 logits 误差0.078125，最大逐请求 KL约0.000163。264个请求步中263个 argmax 一致。

这些结果支持当前容差下的数值对齐，不支持逐位一致或生成文本完全一致的说法。尚未完成通用任务精度评测。

独立虚拟环境现已editable安装开发版；原版源码仍保留作对照。

## 使用

```bash
ssh Nano-vLLM
source /root/autodl-tmp/shared-context/envs/main/bin/activate
export PYTHONPATH=/root/autodl-tmp/shared-context/src/nano-vllm-shared
```

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/root/autodl-tmp/shared-context/models/Qwen3-0.6B-fp16",
    shared_attention_backend="auto",
    enforce_eager=False,
    max_num_seqs=64,
    max_model_len=8192,
    max_num_batched_tokens=8192,
    gpu_memory_utilization=0.65,
)
# 将实际共享前缀的请求交给 generate 或 add_request / step。
# runner.attention_route_counts 可检查是否实际选中了共享路径。
```

FP16目录复用官方模型权重文件，仅提供 FP16 运行配置；原始 BF16 模型目录不变。性能比较的两边均使用相同 FP16 配置。

## 文件与复现

- `nanovllm/layers/shared_prefix.py`：分页 kernel 和稳定合并。
- `nanovllm/engine/shared_plan.py`：共同前缀与路径选择。
- `nanovllm/engine/model_runner.py`：执行元数据与两套 Graph。
- `tests/validate_paged_shared.py`：分页、变长、空段、动态 Graph 和选择器。
- `tests/validate_engine_shared.py`：整模型 logits 与图切换。
- `experiments/bench_integrated.py`：固定负载下原版/off/auto/force对照。
- `experiments/run_integration.sh`：分页与整模型验证、Graph对照矩阵，以及eager回退对照。

Nsight 性能计数器不可用；本轮结论依据实际耗时、路由日志和数值测试，没有声称测得带宽或缓存命中率收益。
