"""Check reordered per-request outputs against fixed histories and exercise chunk fairness."""
import argparse,json,random
from pathlib import Path
import torch
from nanovllm import LLM,SamplingParams
from nanovllm.engine.scheduler import Scheduler,RequestQueueFull
from nanovllm.utils.context import reset_context,get_context


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--profile',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    llm=LLM(args.model,max_num_seqs=64,max_model_len=8192,max_num_batched_tokens=16384,
            gpu_memory_utilization=.65,shared_attention_backend='auto',schedule_cost_profile=args.profile)
    reference={};report=[];runner=llm.model_runner
    # A constant scheduler clock isolates grouping/permutation from GPU verification
    # overhead. Real clock deadlines are exercised by the separate serving replay.
    for policy in ('baseline','fcfs','lpm','cost'):
        llm.config.schedule_policy=policy;sc=Scheduler(llm.config);sc.shared_capable=True
        sc.block_manager.copy_blocks=runner.copy_kv_blocks;llm.scheduler=sc
        llm.config.schedule_policy='baseline'
        parents=[]
        for group in range(2):
            rng=random.Random(91+group)
            parents.append(llm.add_request([rng.randrange(100,10000) for _ in range(4223)],SamplingParams(max_tokens=64,ignore_eos=True)))
        llm.step()
        for rid in parents:llm.pause_request(rid)
        llm.config.schedule_policy=policy
        clock=[0.];sc.clock=lambda:clock[0]
        children=[]
        for i in range(64):children.append(llm.fork_request(parents[i%2],SamplingParams(max_tokens=4,ignore_eos=True),next_token_id=700+i))
        indexes={rid:i for i,rid in enumerate(children)};rows=[];runner.attention_route_counts.clear()
        while not llm.is_finished():
            seqs,prefill=sc.schedule();assert not prefill
            ids,pos=runner.prepare_decode(seqs);logits=runner.run_model(ids,pos,False).float()
            selected=get_context().shared_enabled
            for j,seq in enumerate(seqs):
                key=(indexes[seq.seq_id],seq.num_completion_tokens)
                got=logits[j]
                if policy=='baseline':reference[key]=got.cpu()
                ref=reference[key].to(got.device);rel=((got-ref).norm()/ref.norm()).item()
                kl=(ref.softmax(-1)*(ref.log_softmax(-1)-got.log_softmax(-1))).sum().clamp_min(0).item()
                assert rel<.005 and kl<.001,(policy,key,rel,kl)
                rows.append(dict(request=key[0],token=key[1],relative_l2=rel,kl=kl,shared=selected))
            sc.record_execution(seqs,False,8.)
            sc.postprocess(seqs,[900+indexes[s.seq_id]+s.num_completion_tokens for s in seqs],False)
            reset_context();sc.block_manager.assert_invariants()
        assert len(rows)==256 and len({(r['request'],r['token']) for r in rows})==256
        for rid in parents:llm.cancel_request(rid)
        assert sc.block_manager.stats()['live_pages']==0
        routes=dict(runner.attention_route_counts)
        if policy in ('lpm','cost'):assert routes.get('shared',0)>0,routes
        report.append(dict(policy=policy,routes=routes,rows=rows))
        print(policy,'PASS',routes,flush=True)
    # Real chunked prefill mixed with decode plus admission rollback.
    llm.config.schedule_policy='cost';llm.config.schedule_queue_limit=4
    sc=Scheduler(llm.config);sc.shared_capable=True;sc.block_manager.copy_blocks=runner.copy_kv_blocks;llm.scheduler=sc
    hot=llm.add_request([101]*1151,SamplingParams(max_tokens=16,ignore_eos=True))
    while sc.requests[hot].num_cached_tokens!=len(sc.requests[hot])-1:llm.step()
    cold=llm.add_request([102]*4096,SamplingParams(max_tokens=4,ignore_eos=True))
    llm.save_session(hot,'retained');before=sc.block_manager.stats()['references']
    try:llm.fork_requests(hot,3);assert False
    except RequestQueueFull:pass
    assert sc.block_manager.stats()['references']==before
    phases=[]
    while not llm.is_finished():
        done,work=llm.step();phases.append(work>0);sc.block_manager.assert_invariants()
    llm.close_session('retained');assert sc.block_manager.stats()['live_pages']==0
    assert any(phases) and any(not p for p in phases)
    Path(args.output).write_text(json.dumps(dict(passed=True,permutation=report,mixed_phases=phases,admission_rollback=True),indent=2));llm.exit()

if __name__=='__main__':main()
