"""Causal three-turn replay. Subsequent requests exist only after prior completion + tool delay."""
import argparse,json,time,heapq,statistics
from pathlib import Path
import torch,nanovllm
from nanovllm import LLM,SamplingParams
from nanovllm.engine.sequence import Sequence

VARIANTS={
 'B0':('baseline','off','fifo',False,0),
 'B1':('fcfs','off','fifo',False,0),
 'B2':('lpm','off','fifo',False,0),
 'M':('fcfs','off','lru',True,0),
 'K':('fcfs','auto','fifo',False,0),
 'S':('cost','off','fifo',False,0),
 'Full':('cost','auto','lru',True,0),
 'Tiered':('cost','auto','lru',True,64),
 'Full-no-M':('cost','auto','fifo',False,0),
 'Full-no-K':('cost','off','lru',True,0),
 'Full-no-S':('fcfs','auto','lru',True,0),
}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=VARIANTS,required=True)
 ap.add_argument('--model',required=True);ap.add_argument('--profile',default='results/finalization/cost-profile.json')
 ap.add_argument('--output',required=True);ap.add_argument('--repeats',type=int,default=3)
 ap.add_argument('--sessions',type=int,default=32);ap.add_argument('--scenarios',nargs='+',default=['shared','no_shared']);args=ap.parse_args()
 policy,backend,cache,cow,cpu=VARIANTS[args.variant]
 kwargs=dict(max_num_seqs=64,max_model_len=4096,max_num_batched_tokens=8192,gpu_memory_utilization=.65,
  max_kvcache_blocks=192,schedule_policy=policy,shared_attention_backend=backend,cache_policy=cache,
  cpu_cache_pages=cpu,schedule_prefill_chunk=1024,schedule_prefill_mode='model',schedule_phase_mode='predictive',
  schedule_cost_profile=args.profile)
 if args.variant=='B0':kwargs={k:v for k,v in kwargs.items() if k in ['max_num_seqs','max_model_len','max_num_batched_tokens','gpu_memory_utilization']}
 elif policy!='cost':kwargs.update(schedule_prefill_mode='fixed',schedule_phase_mode='guarded')
 if args.variant=='B0':
  # Capacity-only instrumentation: frozen upstream has no explicit page cap.
  import inspect,textwrap
  import nanovllm.engine.model_runner as module
  source=textwrap.dedent(inspect.getsource(module.ModelRunner.allocate_kv_cache))
  needle='    assert config.num_kvcache_blocks > 0'
  assert needle in source
  source=source.replace(needle,'    config.num_kvcache_blocks = min(config.num_kvcache_blocks, 192)\n'+needle)
  namespace=dict(vars(module));exec(source,namespace)
  module.ModelRunner.allocate_kv_cache=namespace['allocate_kv_cache']
 llm=LLM(args.model,**kwargs);runner=llm.model_runner
 engine_config=getattr(llm,'config',runner.config)
 continuation=llm.tokenizer.encode('\nUser: Summarize the evidence and describe the next validation step.\nAssistant:')
 response=llm.tokenizer.encode('The measurements support a bounded conclusion. Additional independent checks are required before deployment.')
 rows=[]
 for scenario in args.scenarios:
  for repeat in range(-1,args.repeats):
   if True:
    # Fresh scheduler/pool metadata for cold-cache comparisons; physical allocation stays fixed.
    from nanovllm.engine.scheduler import Scheduler
    if hasattr(llm,'transfer_manager'):llm.transfer_manager.close()
    llm.scheduler=Scheduler(engine_config)
    if hasattr(llm.scheduler,'shared_capable'):
     llm.scheduler.shared_capable=runner.shared_supported and (backend!='auto' or runner.shared_auto_calibrated)
     llm.scheduler.block_manager.copy_blocks=runner.copy_kv_blocks
    if cpu:
     from nanovllm.engine.transfer_manager import TransferManager
     llm.transfer_manager=TransferManager(llm.scheduler.block_manager,runner.kv_cache,engine_config,llm.scheduler.cost)
   prompts=[]
   for i in range(args.sessions):
    text=('Shared document. ' if scenario=='shared' else f'Document identity {i}. ')+('Memory management and scheduling require precise ownership, measured latency and reproducible comparisons. '*500)
    head=llm.tokenizer.encode(text)[:1024]
    prompts.append(head+llm.tokenizer.encode(f'\nQuestion {i}: Explain the evaluation.\nAnswer:'))
   records={};events=[];completed={};history={i:list(p) for i,p in enumerate(prompts)}
   now=time.perf_counter();start=now
   for i in range(args.sessions):heapq.heappush(events,(start,i,0))
   post=llm.scheduler.postprocess
   def record_post(seqs,ids,prefill):
    before={s.seq_id:s.num_completion_tokens for s in seqs}
    ids=[response[s.num_completion_tokens%len(response)] for s in seqs]
    post(seqs,ids,prefill);stamp=time.perf_counter()
    for s in seqs:
     if s.num_completion_tokens>before[s.seq_id]:records[s.seq_id]['times'].append(stamp)
   llm.scheduler.postprocess=record_post
   if hasattr(runner,'attention_route_counts'):runner.attention_route_counts.clear()
   prefill_tokens=0
   while events or not llm.is_finished():
    now=time.perf_counter()
    while events and events[0][0]<=now:
     due,sid,turn=heapq.heappop(events)
     if turn:assert due>=completed[(sid,turn-1)]+.02-1e-8
     params=SamplingParams(max_tokens=16,ignore_eos=True)
     if cow and turn:
      rid=llm.fork_session(sid,params,append_token_ids=continuation,output_session_id=sid)
      llm.close_session(sid)
     elif cow:rid=llm.add_request(history[sid],params,session_id=sid)
     else:
      seq=Sequence(history[sid],params);llm.scheduler.add(seq);rid=seq.seq_id
     records[rid]=dict(session=sid,turn=turn,due=due,times=[],input_tokens=len(history[sid]))
    if llm.is_finished():time.sleep(min(.001,max(0.,events[0][0]-time.perf_counter())));continue
    outputs,count=llm.step();prefill_tokens+=max(0,count)
    for rid,tokens in outputs:
     r=records[rid];sid,turn=r['session'],r['turn'];end=time.perf_counter();completed[(sid,turn)]=end
     assert len(tokens)==16
     history[sid].extend(tokens)
     if turn<2:
      history[sid].extend(continuation);heapq.heappush(events,(end+.02,sid,turn+1))
     elif cow:llm.close_session(sid)
   torch.cuda.synchronize();elapsed=time.perf_counter()-start
   llm.scheduler.postprocess=post
   if hasattr(llm,'transfer_manager'):llm.transfer_manager.stream.synchronize();llm.transfer_manager.poll()
   bm=llm.scheduler.block_manager
   if hasattr(bm,'assert_invariants'):bm.assert_invariants();assert not bm.used_block_ids
   ttft=[(r['times'][0]-r['due'])*1000 for r in records.values()]
   maxitl=[max((b-a)*1000 for a,b in zip(r['times'],r['times'][1:])) for r in records.values()]
   good=sum(a<=500 and b<=75 for a,b in zip(ttft,maxitl))
   import numpy as np
   row=dict(variant=args.variant,scenario=scenario,repeat=repeat,elapsed_s=elapsed,requests=len(records),
    output_tokens_per_s=len(records)*16/elapsed,prefill_tokens=prefill_tokens,input_tokens=sum(r['input_tokens'] for r in records.values()),
    ttft_p95_ms=float(np.percentile(ttft,95)),request_max_itl_p95_ms=float(np.percentile(maxitl,95)),
    goodput=good/elapsed,good_requests=good,route_counts=dict(getattr(runner,'attention_route_counts',{})),
    cache_stats=bm.stats() if hasattr(bm,'stats') else {},transfer=llm.transfer_manager.metrics() if cpu else {},
    requests_raw=[dict(session=r['session'],turn=r['turn'],due_s=r['due']-start,token_times_s=[t-start for t in r['times']],input_tokens=r['input_tokens']) for r in records.values()])
   print('WARMUP' if repeat<0 else 'TRIAL',args.variant,scenario,repeat,round(row['output_tokens_per_s'],2),row['good_requests'],flush=True)
   if repeat>=0:
    rows.append(row);Path(args.output).write_text(json.dumps(dict(source=nanovllm.__file__,config=kwargs,trials=rows,
      note='32 sessions x 3 dependent turns; 20ms tool delay; fixed teacher-forced responses for matched work; TTFT<=500ms and max ITL<=75ms. Each scenario has excluded full replay warmup.'),indent=2))
 import atexit
 atexit.unregister(llm.exit)
 llm.exit()
if __name__=='__main__':main()
