"""Matched, fixed-workload serving comparison. No benchmark-specific policy tuning."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import nanovllm
from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.block_manager import BlockManager


SCENARIOS={
    'shared_short':(32,1024,1,0,False),
    'shared_long':(32,4096,1,0,False),
    'small_shared':(8,4096,1,0,False),
    'no_shared':(32,1024,0,0,False),
    'two_groups':(32,2048,2,0,False),
    'shrinking':(33,2048,1,0,True),
    'open_loop':(64,2048,1,80,False),
}


def pct(values):return {str(p):float(np.percentile(values,p)) for p in (50,95,99)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['original','off','auto','force'],required=True)
    ap.add_argument('--model',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--eager',action='store_true');ap.add_argument('--scenarios',nargs='+',default=list(SCENARIOS))
    ap.add_argument('--repeats',type=int,default=3)
    args=ap.parse_args();torch.manual_seed(20261113)
    kwargs=dict(enforce_eager=args.eager,max_num_seqs=64,max_num_batched_tokens=8192,
                max_model_len=8192,gpu_memory_utilization=.65)
    if args.backend!='original':kwargs['shared_attention_backend']=args.backend
    llm=LLM(args.model,**kwargs)
    report={'backend':args.backend,'mode':'eager' if args.eager else 'graph',
            'source':nanovllm.__file__,'config':kwargs,'trials':[],
            'note':'Fixed inputs and output lengths; warm-seeded prefix cache; trial warmup excluded. Not saturation capacity.'}
    for name in args.scenarios:
        n,p,groups,rate,shrinking=SCENARIOS[name]
        for repeat in range(-1,args.repeats):
            assert llm.is_finished()
            rng=random.Random(1200+max(repeat,0))
            torch.manual_seed(1200+max(repeat,0))
            old=llm.scheduler.block_manager
            llm.scheduler.block_manager=BlockManager(len(old.blocks),old.block_size)
            bm=llm.scheduler.block_manager
            heads=[[rng.randrange(100,10000) for _ in range(p)] for _ in range(max(groups,1))]
            for head in heads:
                llm.generate([head+[100]],SamplingParams(max_tokens=1,ignore_eos=True),use_tqdm=False)
            requests=[]
            for i in range(n):
                head=heads[i%groups] if groups else [rng.randrange(100,10000) for _ in range(p)]
                tail=[rng.randrange(100,10000) for _ in range((1,17,63,127,255,257)[i%6])]
                out=32+16*(i%3) if shrinking else 64
                requests.append((head+tail,out))
            records={};stage_times={'prefill':[],'decode':[]};hits=[];preemptions=[]
            allocate=bm.allocate
            def record_allocate(seq,cached):
                hits.append(cached*bm.block_size);return allocate(seq,cached)
            bm.allocate=record_allocate
            post=llm.scheduler.postprocess
            def record_post(seqs,ids,prefill):
                before={s.seq_id:s.num_completion_tokens for s in seqs}
                post(seqs,ids,prefill);now=time.perf_counter()
                for s in seqs:
                    if s.num_completion_tokens>before[s.seq_id]:records[s.seq_id]['times'].append(now)
            llm.scheduler.postprocess=record_post
            preempt=llm.scheduler.preempt
            def record_preempt(seq):preemptions.append(seq.seq_id);return preempt(seq)
            llm.scheduler.preempt=record_preempt
            runner=llm.model_runner
            if hasattr(runner,'attention_route_counts'):
                runner.attention_route_counts.clear();runner.shared_planning_seconds=0
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter();cursor=0
            while cursor<n or not llm.is_finished():
                now=time.perf_counter()
                while cursor<n and (rate==0 or now>=start+cursor/rate):
                    tokens,out=requests[cursor]
                    seq=Sequence(tokens,SamplingParams(max_tokens=out,ignore_eos=True,temperature=.6))
                    records[seq.seq_id]={'arrival':start+(cursor/rate if rate else 0),'expected':out,'times':[]}
                    llm.scheduler.add(seq);cursor+=1
                if llm.is_finished():time.sleep(.0005)
                else:
                    t=time.perf_counter();_,count=llm.step()
                    stage_times['prefill' if count>0 else 'decode'].append(time.perf_counter()-t)
            elapsed=time.perf_counter()-start
            llm.scheduler.postprocess=post;llm.scheduler.preempt=preempt;bm.allocate=allocate
            assert all(len(r['times'])==r['expected'] for r in records.values())
            if repeat<0:
                print('WARMUP',name,flush=True);continue
            ttft=[(r['times'][0]-r['arrival'])*1000 for r in records.values()]
            itl=[(b-a)*1000 for r in records.values() for a,b in zip(r['times'],r['times'][1:])]
            tpot=[(r['times'][-1]-r['times'][0])*1000/(len(r['times'])-1) for r in records.values()]
            row={'scenario':name,'repeat':repeat,'requests':n,'prefix':p,'groups':groups,'arrival_rate':rate,
                 'elapsed_s':elapsed,'output_tokens':sum(r['expected'] for r in records.values()),
                 'output_tokens_per_s':sum(r['expected'] for r in records.values())/elapsed,
                 'ttft_ms':pct(ttft),'itl_ms':pct(itl),'tpot_ms':pct(tpot),
                 'stage_seconds':{k:sum(v) for k,v in stage_times.items()},
                 'steps':{k:len(v) for k,v in stage_times.items()},'cached_tokens':sum(hits),
                 'preemptions':len(preemptions),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                 'route_counts':dict(getattr(runner,'attention_route_counts',{})),
                 'planning_seconds':getattr(runner,'shared_planning_seconds',0),
                 'requests_raw':[{'id':i,'arrival_s':r['arrival']-start,'expected':r['expected'],
                                  'token_times_s':[t-start for t in r['times']]} for i,r in records.items()]}
            report['trials'].append(row);Path(args.output).write_text(json.dumps(report,indent=2))
            print(json.dumps({k:v for k,v in row.items() if k!='requests_raw'}),flush=True)


if __name__=='__main__':main()
