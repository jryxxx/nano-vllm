import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'results/cache';median=statistics.median
rows=sum((json.loads((out/name).read_text())['trials'] for name in ('replay.json','replay-pressure.json')),[])
summary=[]
for scenario in ('pollution','switch','stream','mixed_cost'):
 for pages in (20,32,64):
  for policy in ('fifo','lru','leaf_lru','cost'):
   rr=[r for r in rows if (r['scenario'],r['pages'],r['policy'])==(scenario,pages,policy)]
   assert len(rr)==3
   assert len({r['stats']['reused_tokens'] for r in rr})==1
   summary.append(dict(scenario=scenario,pages=pages,policy=policy,
    tok_s=median(r['output_tokens_per_s'] for r in rr),tok_s_min=min(r['output_tokens_per_s'] for r in rr),tok_s_max=max(r['output_tokens_per_s'] for r in rr),
    tok_s_excluding_first=median(r['output_tokens_per_s'] for r in rr if r['repeat']>0),
    reused_tokens=rr[0]['stats']['reused_tokens'],ttft_p95_ms=median(r['ttft_p95_ms'] for r in rr),
    selector_ms=median(r['stats']['policy_seconds']*1000 for r in rr)))
control=[]
for scenario in ('no_shared','shared_short'):
 vals={}
 for name in ('previous','current'):
  rs=json.loads((out/f'no-pressure-{name}.json').read_text())['trials']
  vals[name]=median(r['output_tokens_per_s'] for r in rs if r['scenario']==scenario)
 control.append(dict(scenario=scenario,**vals,change_percent=100*(vals['current']/vals['previous']-1)))
gpu=json.loads((out/'gpu-validation.json').read_text());checks=[]
for row in gpu['cases']:
 c=row['checks'];checks.append(dict(policy=row['policy'],checks=len(c),max_relative_l2=max(x['relative_l2'] for x in c),max_kl=max(x['kl'] for x in c),
 failed_gate=sum(x['relative_l2']>=.005 or x['kl']>=.001 for x in c),max_previous_fifo_relative_l2=max((x['previous_fifo_relative_l2'] or 0) for x in c)))
resident=json.loads((out/'resident-validation.json').read_text())
result=dict(trials=len(rows),requests=sum(len(r['requests']) for r in rows),summary=summary,no_pressure=control,gpu=checks,
            resident_bitwise_checks=sum(r['bitwise_page_checks'] for r in resident['cases']),
            full_recompute_gate_passed=gpu['full_recompute_gate_passed'])
(out/'summary.json').write_text(json.dumps(result,indent=2))
lines=['# 缓存淘汰阶段实验报告','',
'已实现并接入 FIFO、LRU、依赖感知 LRU 与衰减成本策略；默认仍为 FIFO。缓存复用量有改善，但本轮未证实稳定吞吐收益，成本策略存在热点切换回退。完整重算的严格数值门槛尚未全部通过。','',
'## 实验设置','',
'RTX 4090，Qwen3-0.6B FP16。四种合成文档轨迹、三种逻辑容量（20/32/64 页，每页 256 token）、四种策略、三次重复，共 144 次试验、6912 个请求。单请求串行闭环，每请求固定输出 4 token；底层 GPU KV 分配保持不变。关闭共享注意力，使用 baseline 调度，统一使用上一阶段 Prefill 成本系数。固定衰减半衰期为 32 次分配，未按结果调参。','',
'进程启动时对完整 Prefill 长度预热；缓存命中形成的全部形状未逐一预热，首轮存在编译/执行路径冷启动。下面报告三次中位数，summary.json 同时保留范围及排除首轮的敏感性结果。三次重复不足以把约 1% 波动解释为确定收益。','',
'## 缓存竞争结果','',
'每个单元格为输出 token/s（括号内为累计复用的输入 token）。这不是线上并发 goodput。','',
'| 负载 | 页数 | FIFO | LRU | 依赖感知 LRU | 成本策略 |','|---|---:|---:|---:|---:|---:|']
for sc in ('pollution','switch','stream','mixed_cost'):
 for pages in (20,32,64):
  rs=[r for r in summary if r['scenario']==sc and r['pages']==pages]
  lines.append('| '+sc+' | '+str(pages)+' | '+' | '.join(f"{r['tok_s']:.2f}（{r['reused_tokens']}）" for r in rs)+' |')
lines += ['',
'20 页污染负载中，LRU 的复用量从 24576 提升到 44288 token，但吞吐没有提升。20 页混合长度中成本策略复用了 36096 token，FIFO 为零，吞吐仍无优势。32 页热点切换中成本策略只复用 47360 token，LRU 为 65536，说明历史频次保留了过时热点。依赖感知 LRU 与 LRU 在这些轨迹中命中数相同，不能宣称其独立收益。','',
'选择器开销是整条 48 请求轨迹的累计 CPU 时间，最大约 18.44 ms；它不包含精确前缀索引等全部管理成本。当前实现逐次扫描空闲页，尚未验证大页池扩展性。单请求 TTFT 中位数约 42 ms；命中增加为何未转化为延迟收益仍需进一步分解执行开销，不能据此推断显存带宽或 GPU 利用率。','',
'## 无压力回归','',
'旧版本使用冻结快照 nano-vllm-pre-cache-policy，当前版本使用默认 FIFO；各负载包含一次完整预热和三次重复。两版进程顺序运行，非交错配对，细小差异仍可能含运行漂移。','',
'| 负载 | 旧版 token/s | 当前 token/s | 变化 |','|---|---:|---:|---:|']
for r in control:lines.append(f"| {r['scenario']} | {r['previous']:.2f} | {r['current']:.2f} | {r['change_percent']:+.2f}% |")
lines += ['', '## 正确性验收','',
'- CPU：四种策略共 6000 次随机压力请求，强制旧哈希碰撞、复制失败回滚、反复释放、引用计数和索引一致性通过；调度与 Prefill 模型回归通过。',
f"- GPU：{result['resident_bitwise_checks']} 次驻留页面逐位检查通过；20 页压力下保留会话内容不变，最终所有活跃引用归零。",
'- 既有 COW GPU 回归通过：复制结果逐位一致、快照不变、失败回滚与压力恢复通过，包含共享注意力组合验证。',
'- logits：每种路径 32 请求 × 3 token，与全量重新计算比较。相对 L2 < 0.005、KL < 0.001 门槛保持不变。', '',
'| 路径 | 最大相对 L2 | 最大 KL | 未过门槛 token 数 / 96 |','|---|---:|---:|---:|']
for r in checks:lines.append(f"| {r['policy']} | {r['max_relative_l2']:.6f} | {r['max_kl']:.6f} | {r['failed_gate']} |")
lines += ['',
'当前 FIFO 对上一阶段 FIFO 的最大相对 L2 为 0.004899，通过既定相对误差门槛，但非逐位一致。上一阶段 FIFO 自身对全量重算也有 8 个 token 超过 L2 门槛，表明问题不只出现在本次策略中；这并不能证明所有差异均由旧实现导致。缓存分段与全段执行的 FP16 数值敏感性仍需独立定位。不得把驻留内容检查通过等同于模型质量验收全部通过。','',
'## 当前结论与后续','',
'本阶段交付了可选策略、生命周期保护、可重复脚本和正负对照结果。保留默认 FIFO；LRU 可用于后续压力场景实验；成本策略仍为实验选项。下一步优先处理整段重算数值门槛，并分解缓存命中后的端到端开销；再考虑基于观测收益的策略切换与更快的候选数据结构。CPU offload、多卡、其他模型及 BF16 不属于本阶段完成范围。','',
'源码与方法见 docs/CACHE_POLICY.md；逐请求记录、验证结果、增量补丁和校验清单位于 results/cache。GPU 性能计数器未开启，本报告不包含 DRAM/L2 硬件测量。']
(root/'docs/CACHE_REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(trials=len(rows),resident_checks=result['resident_bitwise_checks'],no_pressure=control),indent=2))
