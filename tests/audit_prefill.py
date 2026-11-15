"""Fresh shape audit: measurements are never added to the fitted profile."""
import argparse,json,time,statistics,random
from pathlib import Path
from types import SimpleNamespace
import torch
from nanovllm import LLM,SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.schedule_cost import ScheduleCost


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--profile',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    llm=LLM(args.model,max_num_seqs=64,max_model_len=8192,max_num_batched_tokens=16384,gpu_memory_utilization=.84,shared_attention_backend='auto')
    cost=ScheduleCost(SimpleNamespace(model=args.model,enforce_eager=False,schedule_cost_profile=args.profile))
    shapes=[([384],[0]),([1280]*2,[1536]*2),([384,896,1792],[256,1280,2048]),
            ([768]*6,[512]*6),([3072],[2560]),([256]*4,[2816]*4)]
    rows=[];rng=random.Random(719)
    for qs,cs in shapes:
        bm=BlockManager(len(llm.scheduler.block_manager.blocks),256);seqs=[]
        for q,c in zip(qs,cs):
            s=Sequence([rng.randrange(100,10000) for _ in range(q+c)],SamplingParams(max_tokens=1));bm.allocate(s,0)
            if c:s.num_scheduled_tokens=c;llm.model_runner.call('run',[s],True)
            s.num_cached_tokens=c;s.num_scheduled_tokens=q;seqs.append(s)
        samples=[]
        for i in range(7):
            t=time.perf_counter();llm.model_runner.call('run',seqs,True);ms=(time.perf_counter()-t)*1000
            if i>=2:samples.append(ms)
        predicted=cost.predict_prefill(qs,cs);actual=statistics.median(samples)
        assert predicted is not None
        rows.append(dict(q=qs,cached=cs,actual_ms=actual,predicted_ms=predicted,relative_error=abs(predicted/actual-1),samples_ms=samples))
        for s in seqs:bm.deallocate(s)
        bm.assert_invariants()
    report=dict(rows=rows,mape=statistics.mean(r['relative_error'] for r in rows),max_relative_error=max(r['relative_error'] for r in rows),profile_updated=False)
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report));llm.exit()

if __name__=='__main__':main()
