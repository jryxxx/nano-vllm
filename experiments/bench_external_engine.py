"""Matched static-batch reference, not attribution of individual engine features."""
import argparse,json,time,random,statistics,atexit
from pathlib import Path
import torch,numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['vllm','nano-off','nano-auto','nano-force'],required=True)
ap.add_argument('--model',required=True);ap.add_argument('--gpu-util',type=float,default=.65);ap.add_argument('--output',required=True);ap.add_argument('--repeats',type=int,default=3);args=ap.parse_args()
is_vllm=args.backend=='vllm'
if is_vllm:
 import vllm
 from vllm import LLM,SamplingParams
 llm=LLM(args.model,dtype='float16',max_num_seqs=64,max_num_batched_tokens=8192,max_model_len=8192,
  gpu_memory_utilization=args.gpu_util,num_gpu_blocks_override=1536,block_size=32,enable_prefix_caching=True,
  enable_chunked_prefill=False,enforce_eager=False,disable_async_output_proc=True,swap_space=0,seed=20261115)
 engine=llm.llm_engine;version=vllm.__version__
else:
 import nanovllm
 from nanovllm import LLM,SamplingParams
 from nanovllm.engine.sequence import Sequence
 from nanovllm.engine.scheduler import Scheduler
 llm=LLM(args.model,max_num_seqs=64,max_num_batched_tokens=8192,max_model_len=8192,
  gpu_memory_utilization=args.gpu_util,max_kvcache_blocks=192,shared_attention_backend=args.backend.split('-')[1])
 version=nanovllm.__file__
rows=[]
for scenario in ['shared','no_shared']:
 for repeat in range(-1,args.repeats):
  if is_vllm:assert engine.reset_prefix_cache()
  else:
   llm.scheduler=Scheduler(llm.config);llm.scheduler.block_manager.copy_blocks=llm.model_runner.copy_kv_blocks
  rng=random.Random(20261115);head=[rng.randrange(100,10000) for _ in range(1024)]
  if scenario=='shared':
   seed=head+[100]
   if is_vllm:llm.generate([{'prompt_token_ids':seed}],SamplingParams(max_tokens=1,ignore_eos=True),use_tqdm=False)
   else:llm.generate([seed],SamplingParams(max_tokens=1,ignore_eos=True),use_tqdm=False)
  prompts=[(head if scenario=='shared' else [rng.randrange(100,10000) for _ in range(1024)])+[100+i]*64 for i in range(32)]
  record={};post=None
  if not is_vllm:
   post=llm.scheduler.postprocess
   def track(seqs,ids,prefill):
    before={s.seq_id:s.num_completion_tokens for s in seqs};post(seqs,ids,prefill);stamp=time.perf_counter()
    for s in seqs:
     if s.num_completion_tokens>before[s.seq_id]:record[s.seq_id].append(stamp)
   llm.scheduler.postprocess=track
  torch.cuda.synchronize();start=time.perf_counter()
  for i,prompt in enumerate(prompts):
   sp=SamplingParams(max_tokens=64,ignore_eos=True,temperature=.6)
   if is_vllm:
    rid=str(i);engine.add_request(rid,{'prompt_token_ids':prompt},sp);record[rid]=[]
   else:
    seq=Sequence(prompt,sp);llm.scheduler.add(seq);record[seq.seq_id]=[]
  if is_vllm:
   while engine.has_unfinished_requests():
    results=engine.step();stamp=time.perf_counter()
    for output in results:
     count=len(output.outputs[0].token_ids)
     record[output.request_id].extend([stamp]*(count-len(record[output.request_id])))
  else:
   while not llm.is_finished():llm.step()
   llm.scheduler.postprocess=post
  torch.cuda.synchronize();elapsed=time.perf_counter()-start
  assert len(record)==32 and all(len(v)==64 for v in record.values())
  ttft=[(v[0]-start)*1000 for v in record.values()]
  itl=[(b-a)*1000 for v in record.values() for a,b in zip(v,v[1:])]
  row=dict(backend=args.backend,scenario=scenario,repeat=repeat,elapsed_s=elapsed,output_tokens_per_s=2048/elapsed,
   ttft_p95_ms=float(np.percentile(ttft,95)),itl_p95_ms=float(np.percentile(itl,95)),
   token_times_s=[[t-start for t in v] for v in record.values()])
  print('WARMUP' if repeat<0 else 'TRIAL',args.backend,scenario,repeat,round(row['output_tokens_per_s'],2),flush=True)
  if repeat>=0:
   rows.append(row);Path(args.output).write_text(json.dumps(dict(version=version,torch=torch.__version__,cuda=torch.version.cuda,trials=rows,
    config=dict(dtype='float16',batch=32,prefix=1024,suffix=64,output=64,kv_capacity_tokens=(1536*32 if is_vllm else llm.config.num_kvcache_blocks*256),gpu_memory_utilization=args.gpu_util,graph=True),
    note='Static simultaneous arrivals; shared-prefix seed excluded. vLLM 0.8.5 V0, 32-token pages versus nano 256-token pages; different Torch/CUDA and attention implementations. No single-feature attribution.'),indent=2))
if not is_vllm:atexit.unregister(llm.exit);llm.exit()
