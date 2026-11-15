import argparse,json
from pathlib import Path
from types import SimpleNamespace
from nanovllm.engine.schedule_cost import ScheduleCost
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence,SequenceStatus
from nanovllm.sampling_params import SamplingParams


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--profile',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    cfg=SimpleNamespace(model=args.model,enforce_eager=False,schedule_cost_profile=args.profile,
        schedule_policy='cost',max_num_seqs=64,max_num_batched_tokens=8192,eos=-1,kvcache_block_size=256,num_kvcache_blocks=639,
        schedule_prefill_chunk=1024,schedule_prefill_mode='model',schedule_phase_mode='predictive',schedule_service_budget_ms=70,
        schedule_prediction_margin=1.1,schedule_planning_margin_ms=2.,schedule_decode_interval_ms=25,schedule_wait_ms=100,
        schedule_max_decode_steps=4,schedule_candidate_limit=512,schedule_max_groups=4,schedule_queue_limit=96,
        shared_min_prefix=1024,shared_min_batch=32,shared_attention_backend='auto')
    cost=ScheduleCost(cfg);profile=json.loads(Path(args.profile).read_text());checks=[]
    for r in profile['prefill']['rows']:
        if r['split']!='holdout':continue
        pred=cost.predict_prefill(r['q'],r['cached']);assert pred is not None and pred>0
        checks.append(dict(q=r['q'],cached=r['cached'],actual_ms=r['ms'],prediction_ms=pred,relative_error=abs(pred/r['ms']-1)))
    assert cost.predict_prefill([8193],[0]) is None
    assert cost.predict_prefill([512],[4096]) is None
    assert cost.predict_prefill([0],[0]) is None
    # No decode work: finish multiple cold prompts in a single bounded step.
    Sequence.block_size=256;sc=Scheduler(cfg);sc.clock=lambda:0.;sc.shared_capable=True
    for i in range(4):sc.add(Sequence([100+i]*1024,SamplingParams(max_tokens=2,ignore_eos=True)))
    seqs,prefill=sc.schedule();assert prefill and len(seqs)==4 and sum(s.num_scheduled_tokens for s in seqs)==4096
    sc.record_execution(seqs,True,40);sc.postprocess(seqs,[200]*4,True);sc.block_manager.assert_invariants()
    for rid in list(sc.requests):sc.cancel(rid)
    # Overdue decode must run before any new predicted prefill.
    sc=Scheduler(cfg);sc.clock=lambda:0.;sc.shared_capable=True
    hot=Sequence([100]*4224,SamplingParams(max_tokens=2));sc.register(hot);sc.block_manager.allocate(hot,0)
    hot.num_cached_tokens=4223;hot.status=SequenceStatus.RUNNING;sc.running.append(hot)
    cold=Sequence([101]*4096,SamplingParams(max_tokens=2));sc.add(cold)
    sc.clock=lambda:.1
    seqs,prefill=sc.schedule();assert not prefill and hot in seqs
    sc.block_manager.assert_invariants()
    for rid in list(sc.requests):sc.cancel(rid)
    # Refresh only the stale tail instead of repeating a whole decode batch.
    sc=Scheduler(cfg);sc.clock=lambda:0.;sc.shared_capable=True;hot_rows=[]
    for i in range(3):
        s=Sequence([100+i]*4224,SamplingParams(max_tokens=2));sc.register(s);sc.block_manager.allocate(s,0)
        s.num_cached_tokens=4223;s.status=SequenceStatus.RUNNING;sc.running.append(s);hot_rows.append(s)
    hot_rows[0].scheduler_service=.04;hot_rows[1].scheduler_service=.04
    sc.add(Sequence([109]*4096));sc.clock=lambda:.05
    seqs,prefill=sc.schedule();assert not prefill and seqs==[hot_rows[2]]
    sc.block_manager.assert_invariants()
    for rid in list(sc.requests):sc.cancel(rid)
    # No prefill profile: old bounded behavior remains available.
    empty_cfg=SimpleNamespace(**vars(cfg));empty_cfg.schedule_cost_profile=''
    sc=Scheduler(empty_cfg);sc.add(Sequence([110]*4096))
    seqs,prefill=sc.schedule();assert prefill and seqs[0].num_scheduled_tokens==1024
    assert sc.metrics['prefill_model_fallback']==1
    for rid in list(sc.requests):sc.cancel(rid)
    result=dict(passed=True,holdout=checks,mape=sum(r['relative_error'] for r in checks)/len(checks),
                batching=True,overdue_decode=True,domain_fallback=True,tail_refresh=True,missing_profile_fallback=True)
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
