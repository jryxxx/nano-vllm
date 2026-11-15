"""Free-running greedy regression; sampler override is test-only, outside model execution."""
import argparse,atexit,json
from pathlib import Path
import torch
from nanovllm import LLM,SamplingParams
from nanovllm.engine.scheduler import Scheduler
ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
llm=LLM(args.model,max_num_seqs=48,max_num_batched_tokens=4096,max_model_len=4096,gpu_memory_utilization=.55,shared_attention_backend='auto')
llm.model_runner.sampler.forward=lambda logits,temperatures: logits.argmax(-1)
head=llm.tokenizer.encode('Shared technical reference: explicit ownership prevents stale cache reuse. '*300)[:1024]
prompts=[head+llm.tokenizer.encode(f'\nQuestion {i}: Explain cache ownership.\nAnswer:') for i in range(33)]
outputs={};routes={}
for backend in ['off','auto']:
 llm.config.shared_attention_backend=backend
 llm.scheduler=Scheduler(llm.config);llm.scheduler.block_manager.copy_blocks=llm.model_runner.copy_kv_blocks
 llm.generate([head+[100]],SamplingParams(max_tokens=1,ignore_eos=True),use_tqdm=False)
 llm.model_runner.attention_route_counts.clear()
 outputs[backend]=[r['token_ids'] for r in llm.generate(prompts,SamplingParams(max_tokens=16,ignore_eos=True),use_tqdm=False)]
 routes[backend]=dict(llm.model_runner.attention_route_counts)
 llm.scheduler.block_manager.assert_invariants();assert llm.cache_stats()['references']==0
assert routes['auto'].get('shared',0)>0
matches=[a==b for a,b in zip(outputs['off'],outputs['auto'])]
report=dict(prompts=33,tokens_per_prompt=16,exact_sequence_matches=sum(matches),token_matches=sum(x==y for a,b in zip(outputs['off'],outputs['auto']) for x,y in zip(a,b)),total_tokens=528,routes=routes,outputs=outputs,note='Free-running greedy regression with test-only argmax sampler. Numeric acceptance is separately gated on identical teacher-forced trajectories.')
Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='outputs'}))
atexit.unregister(llm.exit);llm.exit()
