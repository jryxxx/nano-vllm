import os
import argparse,json,time,statistics
from pathlib import Path
import torch,numpy as np
from nanovllm import LLM,SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.transfer_manager import TransferManager
from experiments.bench_cache import trace

ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['off','auto','always'],required=True)
ap.add_argument('--output',required=True);ap.add_argument('--repeats',type=int,default=3);args=ap.parse_args()
llm=LLM(os.environ.get('NANO_MODEL', '/root/autodl-tmp/shared-context/models/Qwen3-0.6B-fp16'),max_num_seqs=8,max_model_len=4096,
 max_num_batched_tokens=4096,max_kvcache_blocks=20,gpu_memory_utilization=.5,cache_policy='lru',
 cpu_cache_pages=0 if args.mode=='off' else 64,cpu_cache_staging=16,
 cpu_cache_restore='auto' if args.mode=='off' else args.mode,cpu_cache_restore_budget_ms=20.,
 schedule_cost_profile='results/finalization/cost-profile.json')
rows=[]
for scenario in ['pollution','switch','stream','mixed_cost']:
 prompts=trace(llm.tokenizer,scenario)
 for repeat in range(-1,args.repeats):
  if hasattr(llm,'transfer_manager'):llm.transfer_manager.close()
  llm.scheduler=Scheduler(llm.config);bm=llm.scheduler.block_manager;bm.copy_blocks=llm.model_runner.copy_kv_blocks
  tm=None
  if args.mode!='off':tm=TransferManager(bm,llm.model_runner.kv_cache,llm.config,llm.scheduler.cost);llm.transfer_manager=tm
  raw=[];torch.cuda.synchronize();start=time.perf_counter()
  for i,prompt in enumerate(prompts):
   before=bm.counters['reused_tokens'];t=time.perf_counter()
   llm.add_request(prompt,SamplingParams(max_tokens=4,ignore_eos=True));first=None
   while not llm.is_finished():
    llm.step()
    if first is None:first=time.perf_counter()
   raw.append(dict(request=i,ttft_ms=(first-t)*1000,latency_ms=(time.perf_counter()-t)*1000,cached_tokens=bm.counters['reused_tokens']-before))
  torch.cuda.synchronize();elapsed=time.perf_counter()-start
  if tm:tm.poll()
  bm.assert_invariants();assert not bm.used_block_ids
  logical={b.node.uid for b in bm.blocks if b.node is not None}|(set(tm.entries) if tm else set())
  row=dict(mode=args.mode,scenario=scenario,repeat=repeat,output_tokens_per_s=192/elapsed,elapsed_s=elapsed,
   ttft_p95_ms=float(np.percentile([r['ttft_ms'] for r in raw],95)),cache=bm.stats(),transfer=tm.metrics() if tm else {},
   unique_cached_logical_pages=len(logical),gpu_kv_reserved_bytes=llm.model_runner.kv_cache.numel()*llm.model_runner.kv_cache.element_size(),requests=raw)
  print('WARMUP' if repeat<0 else 'TRIAL',args.mode,scenario,repeat,round(row['output_tokens_per_s'],2),row['transfer'].get('restored_pages',0),flush=True)
  if repeat>=0:
   rows.append(row);Path(args.output).write_text(json.dumps(dict(trials=rows,note='Same 20-page GPU pool; CPU variants add 64 pinned pages and 16 GPU staging pages. Full trace warmup excluded per scenario; includes transfer drain. Closed-loop, four output tokens.'),indent=2))
llm.exit()
