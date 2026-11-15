"""Deterministic CPU ownership, rollback and randomized lifetime tests."""
import json
import random
import numpy as np
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.block_manager import BlockManager, CacheCapacityError


def main():
    Sequence.block_size = 4
    bm = BlockManager(96, 4)
    data = np.full((96, 4), -1)
    def copy(jobs):
        for src, dst, count in jobs:
            data[dst, :count] = data[src, :count]
    bm.copy_blocks = copy
    parent = Sequence(list(range(12)))
    bm.allocate(parent, 0)
    for j, page in enumerate(parent.block_table):
        data[page] = np.arange(j * 4, j * 4 + 4)
    parent.num_cached_tokens = 11
    snap = bm.snapshot(parent)
    child = Sequence(list(snap.token_ids))
    bm.clone_many(snap, [child])
    assert bm.blocks[parent.block_table[-1]].ref_count == 3
    bm.ensure_writable(child, 1)
    assert child.block_table[:2] == parent.block_table[:2]
    assert child.block_table[-1] != parent.block_table[-1]
    np.testing.assert_array_equal(data[child.block_table[-1], :3], [8, 9, 10])
    data[child.block_table[-1], 3] = 999
    assert data[parent.block_table[-1], 3] == 11
    bm.assert_invariants()
    # Copy failure must preserve the entire source/target ownership transaction.
    bad = Sequence(list(snap.token_ids)); before = bm.stats()
    def fail(jobs):
        copy(jobs[:1])
        raise RuntimeError('injected copy failure after a partial transfer')
    bm.copy_blocks = fail
    try:
        bm.clone_many(snap, [bad], 'copy')
        assert False
    except RuntimeError:
        pass
    assert not bad.block_table and bm.stats()['live_pages'] == before['live_pages']
    bm.copy_blocks = copy
    cow_bad = Sequence(list(snap.token_ids)); bm.clone_many(snap, [cow_bad])
    table = list(cow_bad.block_table); before = bm.stats()
    bm.copy_blocks = fail
    try:
        bm.ensure_writable(cow_bad, 1)
        assert False
    except RuntimeError:
        pass
    assert cow_bad.block_table == table and bm.stats()['references'] == before['references']
    bm.copy_blocks = copy
    bm.deallocate(cow_bad); bm.deallocate(child); bm.deallocate(parent)
    bm.assert_invariants()
    assert bm.stats()['live_pages'] == 3
    # Full-copy OOM is all-or-nothing, even for bulk branching.
    targets = [Sequence(list(snap.token_ids)) for _ in range(40)]
    try:
        bm.clone_many(snap, targets, 'copy'); assert False
    except CacheCapacityError:
        pass
    assert all(not s.block_table for s in targets)
    rng = random.Random(314159); active = []
    for _ in range(2000):
        if active and (rng.random() < .48 or len(active) > 30):
            s = active.pop(rng.randrange(len(active))); bm.deallocate(s); bm.deallocate(s)
        else:
            s = Sequence(list(snap.token_ids))
            try:
                bm.clone_many(snap, [s], rng.choice(['cow', 'copy']))
                bm.ensure_writable(s, 1)
                data[s.block_table[-1], 3] = rng.randrange(10000)
                active.append(s)
            except CacheCapacityError:
                bm.deallocate(s)
        bm.assert_invariants()
        np.testing.assert_array_equal(data[snap.block_table[-1], :3], [8, 9, 10])
    for s in active: bm.deallocate(s)
    bm.release(snap); bm.release(snap)
    bm.assert_invariants(); assert bm.stats()['live_pages'] == 0
    print(json.dumps(dict(passed=True, randomized_operations=2000, stats=bm.stats())))


if __name__ == '__main__': main()
