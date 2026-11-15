"""CPU cache pressure, exact identity, transactional publication and ownership."""
import json,random
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


def run(policy):
    Sequence.block_size=4
    bm=BlockManager(24,4,policy=policy,half_life=8)
    # Force collisions in the legacy hash to prove exact prefix identities govern hits.
    bm.compute_hash=lambda *args: 7
    def execute(tokens):
        seq=Sequence(tokens);hit=bm.can_allocate(seq);assert hit>=0
        bm.allocate(seq,hit);seq.num_scheduled_tokens=len(seq)-seq.num_cached_tokens
        bm.hash_blocks(seq);seq.num_cached_tokens=len(seq);seq.num_scheduled_tokens=0
        seq.append_token(99)
        bm.assert_invariants();return seq,hit
    first,_=execute([1]*4+[2]*4+[3]);snap=bm.snapshot(first)
    bm.deallocate(first)
    same,hit=execute([1]*4+[2]*4+[4]);assert hit==2;bm.deallocate(same)
    other,hit=execute([8]*4+[2]*4+[4]);assert hit==0;bm.deallocate(other)
    state=[(n.uid,n.last_access,n.frequency) for n in bm.nodes.values()]
    for _ in range(100):bm.can_allocate(Sequence([1]*4+[2]*4+[4]))
    assert state==[(n.uid,n.last_access,n.frequency) for n in bm.nodes.values()]
    # Partial copy failure must leave no published identity at allocated destinations.
    target=Sequence(list(snap.token_ids))
    def fail(jobs):raise RuntimeError('injected failure')
    bm.copy_blocks=fail
    try:bm.clone_many(snap,[target],mode='copy');assert False
    except RuntimeError:pass
    assert not target.block_table
    bm.copy_blocks=lambda jobs:None
    copied=Sequence(list(snap.token_ids));bm.clone_many(snap,[copied],mode='copy')
    bm.assert_invariants();bm.deallocate(copied)
    protected=list(snap.block_table);generation=[bm.blocks[i].generation for i in protected]
    rng=random.Random(20261114)
    for i in range(1500):
        family=rng.randrange(16);depth=rng.randrange(1,5)
        seq,hit=execute([family+100]*4*depth+[i+1000])
        if i%19==0:
            ss=bm.snapshot(seq);child=Sequence(list(ss.token_ids));bm.clone_many(ss,[child])
            bm.ensure_writable(child,1);bm.deallocate(child);bm.release(ss)
        bm.deallocate(seq);bm.assert_invariants()
        assert [bm.blocks[j].generation for j in protected]==generation
    bm.release(snap);bm.assert_invariants();assert not bm.used_block_ids
    assert bm.stats()['evictions']>0
    return bm.stats()

if __name__=='__main__':
    result={p:run(p) for p in ('fifo','lru','leaf_lru','cost')}
    print(json.dumps(dict(passed=True,randomized_requests=6000,policies=result),indent=2))
