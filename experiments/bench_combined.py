"""Six-arm serving ablation; cold parent prefill plus branch lifecycle, same GPU pool."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager

VARIANTS = {
    'copy_off': ('copy','off'), 'copy_auto': ('copy','auto'),
    'cow_off': ('cow','off'), 'cow_auto': ('cow','auto'),
    'eager_tail_off': ('eager_tail','off'), 'eager_tail_auto': ('eager_tail','auto'),
}
SCENARIOS = {
    'short_boundary': dict(cached=256, branches=32, outputs=32),
    'medium': dict(cached=1151, branches=32, outputs=64),
    'long': dict(cached=4223, branches=32, outputs=64),
    'small_batch': dict(cached=4223, branches=8, outputs=64),
    'shrinking': dict(cached=1151, branches=33, outputs=48, shrinking=True),
    'two_groups': dict(cached=1151, branches=32, outputs=64, groups=2),
    'cancel_before_write': dict(cached=4223, branches=32, outputs=64, cancel=24),
}


def fresh_allocator(llm):
    old=llm.scheduler.block_manager
    assert not llm.scheduler.requests and not llm.scheduler.sessions
    old.assert_invariants(); assert old.stats()['live_pages']==0
    bm=BlockManager(len(old.blocks),old.block_size)
    bm.copy_blocks=llm.model_runner.copy_kv_blocks
    llm.scheduler.block_manager=bm
    return bm


def trial(llm, name, variant, repeat):
    spec=SCENARIOS[name]; mode,backend=VARIANTS[variant]
    llm.config.shared_attention_backend=backend
    bm=fresh_allocator(llm); runner=llm.model_runner
    rng=random.Random(301+max(repeat,0)); torch.manual_seed(301+max(repeat,0))
    prompts=[[rng.randrange(100,10000) for _ in range(spec['cached'])] for _ in range(spec.get('groups',1))]
    params=SamplingParams(max_tokens=spec['outputs'],ignore_eos=True,temperature=.6)
    preemptions=[]; original_preempt=llm.scheduler.preempt
    def preempt(seq):
        preemptions.append(seq.seq_id); return original_preempt(seq)
    llm.scheduler.preempt=preempt
    runner.attention_route_counts.clear(); runner.shared_planning_seconds=0
    torch.cuda.synchronize(); start=time.perf_counter()
    parents=[llm.add_request(p,SamplingParams(max_tokens=128,ignore_eos=True)) for p in prompts]
    llm.step()
    for rid in parents: llm.pause_request(rid)
    parent_done=time.perf_counter(); children=[]
    for group,rid in enumerate(parents):
        n=spec['branches']//len(parents)
        children.extend(llm.fork_requests(rid,n,params,next_token_ids=list(range(700+group*100,700+group*100+n)),
                                         cache_mode='copy' if mode=='copy' else 'cow'))
    if mode=='eager_tail':
        # Ablation only: move the SAME per-request COW transaction to fork time.
        # Complete prefix pages remain shared, making attention eligibility identical.
        for rid in children: bm.ensure_writable(llm.scheduler.requests[rid],1)
    if spec.get('shrinking'):
        for i,rid in enumerate(children): llm.scheduler.requests[rid].max_tokens=16+16*(i%3)
    fork_done=time.perf_counter(); live_fork=bm.stats()['live_pages']
    for rid in children[:spec.get('cancel',0)]: llm.cancel_request(rid)
    active=children[spec.get('cancel',0):]
    expected={rid:llm.scheduler.requests[rid].max_tokens for rid in active}
    steps=[]; completed={}; finished_at={}; first=None
    while not llm.is_finished():
        t=time.perf_counter(); done,work=llm.step(); now=time.perf_counter()
        steps.append(dict(ms=(now-t)*1000,batch=abs(work),prefill=work>0))
        if first is None: first=now
        for rid,tokens in done: completed[rid]=tokens; finished_at[rid]=now
    generation_done=time.perf_counter()
    assert set(completed)==set(active)
    assert all(len(completed[rid])==n for rid,n in expected.items())
    assert not preemptions, 'comparison requires equal batches without memory preemption'
    routes=dict(runner.attention_route_counts)
    eligible=(backend=='auto' and mode!='copy' and name in ('medium','long','shrinking'))
    assert (routes.get('shared',0)>0)==eligible,(name,variant,routes)
    for rid in parents: llm.cancel_request(rid)
    end=time.perf_counter(); bm.assert_invariants(); assert bm.stats()['live_pages']==0
    llm.scheduler.preempt=original_preempt
    stats=bm.stats(); output_tokens=sum(expected.values())
    bytes_per_token=2*runner.kv_cache.shape[1]*runner.kv_cache.shape[-2]*runner.kv_cache.shape[-1]*runner.kv_cache.element_size()
    return dict(scenario=name,variant=variant,repeat=repeat,**spec,
        parent_ms=(parent_done-start)*1000,fork_ms=(fork_done-parent_done)*1000,
        branch_first_ms=(first-parent_done)*1000,cold_first_ms=(first-start)*1000,
        branch_total_ms=(generation_done-parent_done)*1000,cold_total_ms=(end-start)*1000,
        cold_tokens_per_s=output_tokens/(end-start),branch_tokens_per_s=output_tokens/(generation_done-parent_done),
        output_tokens=output_tokens,live_after_fork=live_fork,peak_live_pages=stats['peak_live_pages'],
        copy_bytes=stats['copied_tokens']*bytes_per_token,cow_pages=stats['cow_pages'],copy_pages=stats['copy_pages'],
        routes=routes,planning_ms=runner.shared_planning_seconds*1000,steps=steps,
        completion_ms=[(finished_at[r]-parent_done)*1000 for r in active],
        output_sha256=hashlib.sha256(json.dumps([completed[r] for r in active]).encode()).hexdigest(),
        final_live_pages=stats['live_pages'],preemptions=len(preemptions))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--repeats',type=int,default=6);ap.add_argument('--scenarios',nargs='+',default=list(SCENARIOS));args=ap.parse_args()
    llm=LLM(args.model,max_num_seqs=48,max_model_len=8192,max_num_batched_tokens=16384,
            gpu_memory_utilization=.84,shared_attention_backend='auto')
    report=dict(config=dict(model=args.model,gpu=torch.cuda.get_device_name(),dtype=str(llm.model_runner.kv_cache.dtype),
                            pool_pages=len(llm.scheduler.block_manager.blocks),block_size=256,repeats=args.repeats,
                            max_num_seqs=48,gpu_memory_utilization=.84,variants=VARIANTS),trials=[])
    order=list(VARIANTS); random.Random(99).shuffle(order)
    for name in args.scenarios:
        for repeat in range(-1,args.repeats):
            rotated=order[repeat%len(order):]+order[:repeat%len(order)]
            for variant in rotated:
                row=trial(llm,name,variant,repeat)
                if repeat>=0:
                    report['trials'].append(row);Path(args.output).write_text(json.dumps(report,indent=2))
                print(json.dumps({k:v for k,v in row.items() if k not in ('steps','completion_ms')}),flush=True)
    llm.exit()


if __name__=='__main__':main()
