from nanovllm.engine.schedule_cost import runtime_signature
import argparse,json,random,time,statistics
from pathlib import Path
import torch
from nanovllm import LLM,SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.schedule_cost import ScheduleCost


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    llm=LLM(args.model,max_num_seqs=64,max_model_len=8192,max_num_batched_tokens=16384,
            gpu_memory_utilization=.65,shared_attention_backend='auto')
    rows=[];rng=random.Random(555)
    for context in (1151,4223):
        for groups,batch in ((1,8),(1,16),(1,32),(1,64),(2,32),(2,64)):
            for backend in ('off','auto'):
                llm.config.shared_attention_backend=backend
                bm=BlockManager(len(llm.scheduler.block_manager.blocks),256);bm.copy_blocks=llm.model_runner.copy_kv_blocks;llm.scheduler.block_manager=bm
                parents=[llm.add_request([rng.randrange(100,10000) for _ in range(context)],SamplingParams(max_tokens=64,ignore_eos=True)) for _ in range(groups)]
                llm.step()
                for rid in parents:llm.pause_request(rid)
                for rid in parents:llm.fork_requests(rid,batch//groups,SamplingParams(max_tokens=32,ignore_eos=True))
                durations=[]
                features=ScheduleCost.features(list(llm.scheduler.running),llm.config,True)
                for i in range(32):
                    t=time.perf_counter();llm.step();elapsed=(time.perf_counter()-t)*1000
                    if i>=8:durations.append(elapsed)
                rows.append(dict(features,ms=statistics.median(durations),samples_ms=durations))
                for rid in parents:llm.cancel_request(rid)
                bm.assert_invariants();assert bm.stats()['live_pages']==0
                print(context,groups,batch,backend,rows[-1]['ms'],flush=True)
    # Duplicate off/auto fallback observations are combined.
    merged={}
    for row in rows:
        key=(row['batch'],row['context'],row['common'],row['shared'])
        if key not in merged:merged[key]=dict(row)
        else:merged[key]['samples_ms']+=row['samples_ms']
    for row in merged.values():row['ms']=statistics.median(row['samples_ms'])
    Path(args.output).write_text(json.dumps(dict(runtime_signature=runtime_signature(llm.config),model=args.model,eager=False,gpu=torch.cuda.get_device_name(),decode=list(merged.values())),indent=2))
    llm.exit()

if __name__=='__main__':main()
