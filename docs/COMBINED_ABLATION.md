# 会话 COW × 共享前缀注意力：端到端对照与消融

本阶段复用已验证的远程引擎，通过同一个配置对象切换现有 attention planner，未修改生产引擎的选择阈值。新增可复现实验和组合正确性测试。

## 六组定义与可识别性

| 实验组 | 完整前缀物理页 | 部分尾页 | 注意力 |
|---|---|---|---|
| copy_off | 每分支复制 | 分支时复制 | 原分页注意力 |
| copy_auto | 每分支复制 | 分支时复制 | 开启 auto，因物理前缀不同回退 |
| cow_off | 共享 | 首次写入时复制 | 原分页注意力 |
| cow_auto | 共享 | 首次写入时复制 | auto 选择共享前缀注意力 |
| eager_tail_off | 共享 | 分支时立即物化私有尾页 | 原分页注意力 |
| eager_tail_auto | 共享 | 分支时立即物化私有尾页 | auto 选择共享前缀注意力 |

当前共享注意力要求各请求的物理前缀页 ID 相同。完整复制 + auto 因此只是负对照，不能称为“共享注意力单独生效”。eager_tail 复用同一个 COW 复制事务，只把执行时机前移；它是控制实验中的“完整前缀共享 + 立即复制尾块”，不是新增的公共缓存后端。

可归因对比：copy_off → eager_tail_off 衡量完整前缀共享及其伴随的数据复制/访存变化；eager_tail_off → eager_tail_auto 衡量同样页面布局上的共享注意力；eager_tail → cow 衡量尾块延迟物化；cow_off → cow_auto 衡量共享注意力在会话 COW 上的增量。不能把各项加速百分比直接相加。

## 方法

RTX 4090，Qwen3-0.6B FP16；单 GPU、CUDA Graph；256 token/页；所有组使用同一预分配池（639 页），普通和共享 Graph 均提前捕获。每场景每组 1 次预热不计入，6 次测量；六组轮转顺序，每组在各次序位置出现一次。

输入是固定长度的合成 token 负载。每次测量重建空的逻辑块管理器，固定同一 repeat 的 prompt、pending token、随机种子和生成长度。冷会话总耗时包含父请求 prefill/一次输出、暂停、分支构造/复制、子请求生成和父请求回收；模型加载/编译/Graph 捕获排除。分支耗时从父请求完成后开始，包含首次写入 COW。吞吐分子仅计子请求输出 token，父请求准备成本在分母中。

真实采样参与计时，ignore_eos 固定输出长度；不同注意力数值路径可能产生不同 token，因此另外使用完全相同的 teacher-forced 历史做数值验证。每轮记录路由计数、复制量、峰值活跃页、逐步耗时、结束引用和抢占次数。所有计时样本均无抢占、最终引用归零。小幅差异不作显著性结论；原始数据保留每次测量和配对范围。

## 冷会话端到端结果

| 场景 | copy_off ms | cow_off ms | cow_auto ms | 组合相对复制吞吐 | 注意力相对 COW 增量 |
|---|---:|---:|---:|---:|---:|
| short_boundary | 161.53 | 142.62 | 143.15 | +12.84% | -0.37% |
| medium | 523.43 | 362.25 | 334.51 | +56.48% | +8.29% |
| long | 1331.58 | 703.74 | 539.49 | +146.82% | +30.45% |
| small_batch | 536.22 | 384.34 | 384.17 | +39.58% | +0.04% |
| shrinking | 344.88 | 276.58 | 256.66 | +34.38% | +7.76% |
| two_groups | 523.48 | 362.33 | 367.28 | +42.53% | -1.35% |
| cancel_before_write | 574.23 | 395.08 | 395.39 | +45.23% | -0.08% |

场景：short_boundary=256 token/32 分支/32 输出；medium=1151/32/64；long=4223/32/64；small_batch=4223/8/64；shrinking=1151/33、输出 16/32/48 混合；two_groups=两个独立父会话各 1151 token、各 16 分支/64 输出；cancel_before_write=4223/32，首次写入前取消 24 个，剩余 8 个输出 64 token。

## 长上下文完整消融（4223 token × 32 分支）

| 组 | 冷会话总 ms | 分支总 ms | 分支到首 token ms | 峰值活跃页 | 复制 GiB | 共享 step / 全部 decode step |
|---|---:|---:|---:|---:|---:|---:|
| copy_off | 1331.58 | 1280.71 | 68.54 | 561 | 14.434 | 0 / 384 |
| copy_auto | 1332.43 | 1281.57 | 68.55 | 561 | 14.434 | 0 / 384 |
| cow_off | 703.74 | 652.89 | 28.08 | 49 | 0.434 | 0 / 384 |
| cow_auto | 539.49 | 488.65 | 25.53 | 49 | 0.434 | 384 / 384 |
| eager_tail_off | 703.81 | 652.99 | 28.12 | 49 | 0.434 | 0 / 384 |
| eager_tail_auto | 539.80 | 488.68 | 25.48 | 49 | 0.434 | 384 / 384 |

## 尾块延迟物化：首次写入前取消 24/32 分支

| 组 | 分支构造 ms | 分支到首 token ms | 冷会话总 ms | 复制 GiB | 峰值活跃页 |
|---|---:|---:|---:|---:|---:|
| copy_off | 49.20 | 57.20 | 574.23 | 14.434 | 561 |
| eager_tail_off | 18.18 | 23.76 | 398.60 | 0.434 | 49 |
| cow_off | 13.10 | 19.98 | 395.08 | 0.109 | 25 |

该场景用于隔离按需复制的实际价值：cow 不为取消的分支复制尾页，eager_tail 已支付其复制成本。其余立即开始解码的场景，两者应复制相同数量的尾页；构造耗时减少不等于首 token 或总耗时必然同比改善。

## 路由与正确性

- 18 组组合验证、702 个模型 step，涵盖长上下文 64 token 连续生成、33→22→11 缩批（16/32/48 输出）和 0/1/17/257 token 变长续接。全程使用原注意力的实验组与完整复制基线 logits 逐位一致。
- 包含共享注意力及缩批后回退的路径：最大相对 L2 0.002048，最大 KL 0.000050，最大绝对 logits 差 0.048828；argmax 19378/19392 一致。沿用 L2 < 0.005、KL < 0.001 的门槛，不声称组合路径位级等价。
- 缩批回退时，KV 历史已包含此前共享注意力的舍入差异，因此仍按整体 L2/KL 门槛验证，不要求回退 step 恢复位级一致。
- medium/long 的 cow_auto 与 eager_tail_auto 实际命中共享路径；copy_auto 因 no_common_prefix 回退。short_boundary、small_batch、two_groups 分别验证短前缀、小批量和无共同前缀回退。shrinking 在降至 32 以下后回退；取消场景只剩 8 个活跃分支，也回退。

## 结论边界

- 物理前缀共享是当前共享注意力算子的前提，二者不是完全独立的开关。页面共享的收益不应全部归因于延迟尾块 COW。
- 此处是固定并发与输出长度的端到端对照，不是开放环到达流量、生产 P99 SLA 或最大饱和容量测试。
- 活跃页数是预分配池中的占用；所有组的 GPU 池大小一致，不等于 CUDA reserved memory 同比下降。复制字节按有效 token 的单份 KV 数据量计算，不是硬件 DRAM 事务计数。
- GPU Performance Counters 未启用，不对缓存命中率或硬件访存量作实测归因。

## 产物与复现

`experiments/bench_combined.py`：六组计时实验；`tests/validate_combined.py`：teacher-forced 组合验证；`results/combined/benchmark.json`：原始计时；`validation.json`：逐步数值；`summary.json`：中位数、范围及路由计数。

```bash
bash /root/autodl-tmp/shared-context/run_cow_env.sh python experiments/bench_combined.py \
  --model /root/autodl-tmp/shared-context/models/Qwen3-0.6B-fp16 \
  --output results/combined/benchmark.json
bash /root/autodl-tmp/shared-context/run_cow_env.sh python tests/validate_combined.py \
  --model /root/autodl-tmp/shared-context/models/Qwen3-0.6B-fp16 \
  --output results/combined/validation.json
```
