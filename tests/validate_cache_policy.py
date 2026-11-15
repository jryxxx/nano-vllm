"""Victim semantics: leaf protection, decay and duplicate physical pages."""
import json
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
Sequence.block_size=4
bm=BlockManager(2,4,'lru',half_life=8)
seq=Sequence([1]*4+[2]*4)
bm.allocate(seq,0);seq.num_scheduled_tokens=8;bm.hash_blocks(seq)
a,b=seq.block_table;parent=bm.blocks[a].node;child=bm.blocks[b].node
bm.deallocate(seq);parent.last_access=0;child.last_access=1
assert bm._victim()==a
bm.policy='leaf_lru';assert bm._victim()==b
bm.policy='cost';assert bm._victim()==b
bm.epoch=1;before=bm._score(child);bm.epoch+=8
assert abs(bm._score(child)/before-.5)<1e-12
bm._touch(child,reuse=True);assert bm._score(child)>before
bm.assert_invariants()
# Reusing one physical copy must preserve lookup through the other copy.
bm=BlockManager(4,4,'lru');seq=Sequence([7]*4+[8]);bm.allocate(seq,0)
seq.num_scheduled_tokens=5;bm.hash_blocks(seq);seq.num_scheduled_tokens=0;seq.num_cached_tokens=5;seq.append_token(9)
bm.copy_blocks=lambda jobs:None
other=Sequence(list(seq.token_ids));bm.clone_many(seq,[other],mode='copy')
identity=bm.blocks[seq.block_table[0]].node;assert len(identity.locations)==2
bm.deallocate(seq);bm.deallocate(other)
# Partial pages are reclaimed first, then one redundant full page.
allocated=[bm._allocate_block() for _ in range(3)]
assert len(identity.locations)==1 and bm._lookup(identity)>=0
bm._release_ids(allocated);bm.assert_invariants()
print(json.dumps(dict(passed=True,leaf_protection=True,decay=True,duplicate_reclamation=True)))
