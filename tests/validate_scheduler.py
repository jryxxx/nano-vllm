import json
from types import SimpleNamespace
from nanovllm.engine.scheduler import Scheduler,RequestQueueFull
from nanovllm.engine.sequence import Sequence,SequenceStatus
from nanovllm.sampling_params import SamplingParams


def config(policy='fcfs',**kw):
    d=dict(model='test',max_num_seqs=4,max_num_batched_tokens=64,eos=-1,kvcache_block_size=4,
           num_kvcache_blocks=64,schedule_policy=policy,schedule_cost_profile='',enforce_eager=False,
           schedule_prefill_chunk=4,schedule_prefill_target_ms=8,schedule_decode_interval_ms=25,
           schedule_wait_ms=100,schedule_max_decode_steps=4,schedule_candidate_limit=64,schedule_max_groups=4,
           schedule_queue_limit=0,shared_min_prefix=4,shared_min_batch=2,shared_attention_backend='auto')
    d.update(kw);return SimpleNamespace(**d)


def main():
    Sequence.block_size=4; time=[0.]; sc=Scheduler(config());sc.clock=lambda:time[0]
    def step():
        seqs,prefill=sc.schedule();time[0]+=.01
        sc.record_execution(seqs,prefill,10)
        sc.postprocess(seqs,[10]*len(seqs),prefill);sc.block_manager.assert_invariants()
        return [s.seq_id for s in seqs],prefill
    hot=Sequence([1,2,3,4],SamplingParams(max_tokens=100,ignore_eos=True));sc.add(hot);step()
    cold=Sequence(list(range(24)),SamplingParams(max_tokens=2,ignore_eos=True));sc.add(cold)
    phases=[]
    for _ in range(16):
        if sc.is_finished():break
        ids,p=step();phases.append(p)
    assert all(not(a and b) for a,b in zip(phases,phases[1:])),phases
    assert cold.is_finished and hot.num_completion_tokens>5
    for rid in list(sc.requests):sc.cancel(rid)
    sc.block_manager.assert_invariants();assert sc.block_manager.stats()['live_pages']==0
    # Fair rotation across more runnable requests than the batch limit.
    sc=Scheduler(config(max_num_seqs=2));sc.clock=lambda:time[0]
    children=[]
    for i in range(5):
        seq=Sequence([i],SamplingParams(max_tokens=8));sc.add(seq);sc.waiting.remove(seq)
        sc.block_manager.allocate(seq,0);seq.status=SequenceStatus.RUNNING;sc.running.append(seq);children.append(seq)
    seen=set()
    for _ in range(3):
        seqs,p=sc.schedule();seen.update(s.seq_id for s in seqs);time[0]+=.001
        sc.record_execution(seqs,p,1);sc.postprocess(seqs,[10]*len(seqs),p)
    assert len(seen)==5
    # Cost selects a profitable group, then honors an overdue outsider.
    sc=Scheduler(config('cost'));sc.clock=lambda:time[0];sc.shared_capable=True
    for i,page in enumerate([0,1,0,1]):
        seq=Sequence([1]*5);seq.block_table=[page,2+i];seq.num_cached_tokens=4
        sc.register(seq);sc.running.append(seq)
    sc.cost.rows=[dict(batch=4,context=5,common=False,shared=False,ms=10),
                  dict(batch=2,context=5,common=True,shared=True,ms=3)]
    sc._order_decode(time[0]);assert sc._decode_limit==2
    time[0]+=.03;sc._order_decode(time[0]);assert sc._decode_limit==4
    sc=Scheduler(config(schedule_queue_limit=1));sc.add(Sequence([1]))
    try:sc.add(Sequence([2]));assert False
    except RequestQueueFull:pass
    assert len(sc.requests)==1
    print(json.dumps(dict(passed=True,phase_sequence=phases,rotation=True,cost_group=True,deadline_override=True,admission=True)))

if __name__=='__main__':main()
