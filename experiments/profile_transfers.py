import os
"""Controlled background spill alongside live Decode; exports actual CUDA timeline."""
import json
from pathlib import Path
import torch
from torch.profiler import profile,ProfilerActivity,record_function
from nanovllm import LLM,SamplingParams

llm=LLM(os.environ.get('NANO_MODEL', '/root/autodl-tmp/shared-context/models/Qwen3-0.6B-fp16'),max_num_seqs=8,max_model_len=4096,
 max_num_batched_tokens=4096,max_kvcache_blocks=32,gpu_memory_utilization=.5,
 cpu_cache_pages=32,cpu_cache_staging=16,cache_policy='lru',
 schedule_cost_profile='results/finalization/cost-profile.json')
llm.generate([[123]*3072+[100]],SamplingParams(max_tokens=1,ignore_eos=True),use_tqdm=False)
rid=llm.add_request([500]*128,SamplingParams(max_tokens=32,ignore_eos=True))
llm.step()
bm=llm.scheduler.block_manager;tm=llm.transfer_manager
pages=[b for b in bm.blocks if b.ref_count==0 and b.node is not None]
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
 with record_function('background_spill'):
  for b in pages:tm.spill(b)
 while not llm.is_finished():
  with record_function('decode_step'):llm.step()
 tm.stream.synchronize();tm.poll()
prof.export_chrome_trace('results/finalization/transfer-trace.json')
trace=json.loads(Path('results/finalization/transfer-trace.json').read_text())['traceEvents']
decode=[(e['ts'],e['ts']+e['dur']) for e in trace if e.get('name')=='decode_step' and 'dur' in e]
kernels=[e for e in trace if e.get('cat')=='kernel' and 'dur' in e and any(a<=e['ts']<b for a,b in decode)]
copies=[e for e in trace if 'Memcpy DtoH' in e.get('name','') and 'dur' in e]
intervals=[]
for c in copies:
 for k in kernels:
  a=max(c['ts'],k['ts']);b=min(c['ts']+c['dur'],k['ts']+k['dur'])
  if b>a:intervals.append((a,b))
merged=[]
for a,b in sorted(intervals):
 if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
 else:merged.append((a,b))
result=dict(dtoh_events=len(copies),decode_kernel_events=len(kernels),overlap_us=sum(b-a for a,b in merged),
 transfer=tm.metrics(),note='Controlled background spill of immutable free pages while a live request decodes. Not a serving throughput claim.')
Path('results/finalization/transfer-overlap.json').write_text(json.dumps(result,indent=2));print(result)
bm.assert_invariants();llm.exit()
