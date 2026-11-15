import argparse
import json
import random
from pathlib import Path
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager, CacheCapacityError
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import get_context, reset_context


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True)
    ap.add_argument('--output', required=True); ap.add_argument('--backend', default='off')
    ap.add_argument('--eager', action='store_true'); args = ap.parse_args()
    llm = LLM(args.model, enforce_eager=args.eager, shared_attention_backend=args.backend,
              max_num_seqs=48, max_model_len=8192, max_num_batched_tokens=8192, gpu_memory_utilization=.55)
    runner, scheduler = llm.model_runner, llm.scheduler
    bm = scheduler.block_manager
    rng = random.Random(42)
    rows = []
    def clean():
        for rid in list(scheduler.requests): llm.cancel_request(rid)
        for sid in list(scheduler.sessions): llm.close_session(sid)
        bm.assert_invariants(); assert bm.stats()['live_pages'] == 0
    def prepare_parent(p):
        rid = llm.add_request([rng.randrange(100, 10000) for _ in range(p)], SamplingParams(max_tokens=64, ignore_eos=True))
        llm.step(); llm.pause_request(rid); return rid
    for p, extension in [(255,0),(256,0),(257,0),(1024,0),(1151,0),(255,3),(1151,300)]:
        rid = prepare_parent(p); parent = scheduler.requests[rid]
        llm.save_session(rid, 'root'); snap = scheduler.sessions['root']
        frozen = [(page, min(256,p-j*256), runner.kv_cache[:,:,page,:min(256,p-j*256)].clone())
                  for j,page in enumerate(snap.block_table)]
        pairs = []
        for token in [301, 302]:
            pair = []
            for mode in ['cow','copy']:
                pair.append(llm.fork_session('root', SamplingParams(max_tokens=4,ignore_eos=True),
                            next_token_id=token, append_token_ids=[303]*extension, cache_mode=mode))
            pairs.append(pair)
        # Independent oracle: recompute each branch history without any cache reuse.
        references = {}
        for a, b in pairs:
            fresh = Sequence(list(scheduler.requests[a].token_ids))
            bm.allocate(fresh, 0)
            fresh.num_scheduled_tokens = len(fresh)
            inputs, positions = runner.prepare_prefill([fresh])
            references[a] = runner.run_model(inputs, positions, True).float().clone()[0]
            reset_context(); bm.deallocate(fresh)
        llm.cancel_request(rid)
        fresh_rel = 0.
        peak = 0.; steps = 0
        while not llm.is_finished():
            seqs,prefill = scheduler.schedule()
            inputs,positions = runner.prepare_prefill(seqs) if prefill else runner.prepare_decode(seqs)
            logits = runner.run_model(inputs,positions,prefill).float()
            indices = {s.seq_id:i for i,s in enumerate(seqs)}
            for a,b in pairs:
                if a in indices and b in indices:
                    x,y = logits[indices[a]],logits[indices[b]]
                    if steps == 0:
                        reference = references[a]
                        relative = ((x-reference).norm()/reference.norm()).item()
                        assert relative < .005, relative
                        fresh_rel = max(fresh_rel, relative)
                    peak = max(peak,(x-y).abs().max().item())
                    torch.testing.assert_close(x,y,atol=.015625,rtol=.002)
                    assert (x-y).norm()/x.norm() < .0005
            scheduler.postprocess(seqs,[400+steps]*len(seqs),prefill); reset_context()
            bm.assert_invariants(); steps += 1
            for page,valid,value in frozen:
                assert torch.equal(runner.kv_cache[:,:,page,:valid],value), 'snapshot modified by a child'
        rows.append(dict(cached=p,extension=extension,steps=steps,cow_copy_max_abs=peak,snapshot_bitwise_unchanged=True,fresh_recompute_relative_l2=fresh_rel))
        clean()
    # Completion saves a snapshot; parent can disappear and a descendant can branch again.
    rid=llm.add_request([600]*257,SamplingParams(max_tokens=1,ignore_eos=True),session_id='finished')
    llm.step(); assert rid not in scheduler.requests and 'finished' in scheduler.sessions
    child=llm.fork_session('finished',SamplingParams(max_tokens=3,ignore_eos=True))
    llm.step(); llm.save_session(child,'generation2'); llm.pause_request(child)
    grandchild=llm.fork_session('generation2',SamplingParams(max_tokens=2,ignore_eos=True))
    llm.cancel_request(child); llm.close_session('finished'); llm.close_session('generation2')
    while not llm.is_finished(): llm.step(); bm.assert_invariants()
    assert not llm.cancel_request(grandchild) and not llm.close_session('generation2')
    clean()
    rid=prepare_parent(257); llm.save_session(rid,'overwrite')
    llm.resume_request(rid); llm.step(); llm.pause_request(rid)
    llm.save_session(rid,'overwrite'); bm.assert_invariants(); clean()
    # Actual copied values are bitwise identical; unwritten slots are not copied.
    rid=prepare_parent(1151); parent=scheduler.requests[rid]
    child=llm.fork_request(rid,cache_mode='copy'); c=scheduler.requests[child]
    for j,(src,dst) in enumerate(zip(parent.block_table,c.block_table)):
        n=min(256,1151-j*256)
        assert torch.equal(runner.kv_cache[:,:,src,:n],runner.kv_cache[:,:,dst,:n])
    clean()
    # Failure injection exercises scheduler rollback and subsequent successful reuse.
    rid=prepare_parent(257); child=llm.fork_request(rid)
    original=bm.copy_blocks
    def fail(jobs):
        original(jobs)
        raise RuntimeError('injected completed-copy error')
    bm.copy_blocks=fail
    try:
        llm.step(); assert False
    except RuntimeError as e:
        assert 'injected' in str(e)
    bm.assert_invariants(); assert scheduler.requests[child].num_scheduled_tokens==0
    bm.copy_blocks=original; llm.step(); clean()
    # Shrink only the logical allocator; use the same preallocated GPU pool.
    full=bm; bm=BlockManager(3,256); bm.copy_blocks=runner.copy_kv_blocks; scheduler.block_manager=bm
    rid=prepare_parent(257); llm.save_session(rid,'pressure'); llm.cancel_request(rid)
    try:
        llm.fork_session('pressure',cache_mode='copy'); assert False
    except CacheCapacityError:
        pass
    child=llm.fork_session('pressure',SamplingParams(max_tokens=2,ignore_eos=True)); llm.step()
    bm.assert_invariants(); clean()
    rid=prepare_parent(513); llm.save_session(rid,'full'); llm.cancel_request(rid)
    child=llm.fork_session('full',SamplingParams(max_tokens=2,ignore_eos=True))
    try:
        llm.step(); assert False
    except CacheCapacityError:
        pass
    bm.assert_invariants(); llm.close_session('full')
    while not llm.is_finished(): llm.step()
    clean(); scheduler.block_manager=full; bm=full
    # Shared-prefix graph route still operates with COW-created partial tails.
    if args.backend=='auto' and not args.eager:
        rid=prepare_parent(1151)
        ids=llm.fork_requests(rid,32,SamplingParams(max_tokens=2,ignore_eos=True),next_token_ids=list(range(700,732)))
        seqs,prefill=scheduler.schedule(); assert not prefill
        inputs,positions=runner.prepare_decode(seqs); ctx=get_context(); assert ctx.shared_enabled
        got=runner.run_model(inputs,positions,False).float()
        ctx.shared_enabled=False; ref=runner.run_model(inputs,positions,False).float()
        rel=((got-ref).norm()/ref.norm()).item(); assert rel<.005
        scheduler.postprocess(seqs,[800]*len(seqs),False); reset_context(); clean()
        rows.append(dict(shared_cow_batch=32,relative_l2=rel))
    report=dict(passed=True,backend=args.backend,eager=args.eager,cases=rows,
                lifecycle=True,copy_bitwise=True,failure_rollback=True,pressure_recovery=True,final_stats=bm.stats())
    Path(args.output).write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)
    llm.exit()


if __name__=='__main__': main()
