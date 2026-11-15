from collections import deque
from collections import Counter
from dataclasses import dataclass
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class CacheCapacityError(RuntimeError):
    """Insufficient evictable KV pages; retained sessions must be released."""


@dataclass(frozen=True)
class CacheSnapshot:
    token_ids: tuple
    num_cached_tokens: int
    block_table: tuple


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.owners = {}
        self.copy_blocks = None
        self.counters = dict(cow_pages=0, copy_pages=0, copied_tokens=0, peak_live_pages=0)

    def _track(self, owner):
        self.owners[id(owner)] = owner
        self.counters['peak_live_pages'] = max(self.counters['peak_live_pages'], len(self.used_block_ids))

    def _release_ids(self, ids):
        for block_id in ids:
            block = self.blocks[block_id]
            assert block.ref_count > 0
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

    def release(self, owner):
        if self.owners.pop(id(owner), None) is not None:
            self._release_ids(owner.block_table)

    def _copy(self, jobs):
        # Synchronous ownership transaction: source references and newly allocated
        # destinations stay live until the GPU copy has completed, including errors.
        if jobs:
            if self.copy_blocks is None:
                raise RuntimeError('KV copy callback is not installed')
            self.copy_blocks(jobs)

    def snapshot(self, seq):
        self._check_source(seq)
        n = (seq.num_cached_tokens + self.block_size - 1) // self.block_size
        snap = CacheSnapshot(tuple(seq.token_ids), seq.num_cached_tokens, tuple(seq.block_table[:n]))
        for i in snap.block_table:
            self.blocks[i].ref_count += 1
        self._track(snap)
        return snap

    def _check_source(self, source):
        if id(source) not in self.owners:
            raise ValueError('source no longer owns KV cache')
        if getattr(source, 'num_scheduled_tokens', 0) or source.num_cached_tokens != len(source.token_ids) - 1:
            raise ValueError('branching requires a completed step with exactly one pending token')

    def clone_many(self, source, targets, mode='cow'):
        self._check_source(source)
        if mode not in ('cow', 'copy'):
            raise ValueError('cache_mode must be cow or copy')
        n = (source.num_cached_tokens + self.block_size - 1) // self.block_size
        source_ids = source.block_table[:n]
        if any(t.block_table for t in targets):
            raise ValueError('clone target already has cache')
        if mode == 'copy' and len(self.free_block_ids) < n * len(targets):
            raise CacheCapacityError('full-copy fork does not fit in KV cache')
        allocated, tables, jobs = [], [], []
        try:
            for target in targets:
                table = []
                for j, src in enumerate(source_ids):
                    if mode == 'cow':
                        dst = src
                    else:
                        dst = self._allocate_block()
                        allocated.append(dst)
                        valid = min(self.block_size, source.num_cached_tokens - j * self.block_size)
                        jobs.append((src, dst, valid))
                        if valid == self.block_size:
                            self.blocks[dst].update(self.blocks[src].hash, list(self.blocks[src].token_ids))
                    table.append(dst)
                tables.append(table)
            self._copy(jobs)
        except BaseException:
            self._release_ids(allocated)
            raise
        for target, table in zip(targets, tables):
            if mode == 'cow':
                for i in table:
                    self.blocks[i].ref_count += 1
            target.block_table = table
            target.num_cached_tokens = source.num_cached_tokens
            self._track(target)
        self.counters['copy_pages'] += len(jobs)
        self.counters['copied_tokens'] += sum(j[2] for j in jobs)

    def write_plan(self, seq, count):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + count + self.block_size - 1) // self.block_size
        shared = [i for i in range(start, min(end, len(seq.block_table)))
                  if self.blocks[seq.block_table[i]].ref_count > 1]
        missing = max(0, end - len(seq.block_table))
        return shared, missing

    def can_write(self, seq, count):
        shared, missing = self.write_plan(seq, count)
        return len(self.free_block_ids) >= len(shared) + missing

    def ensure_writable(self, seq, count):
        shared, missing = self.write_plan(seq, count)
        if len(self.free_block_ids) < len(shared) + missing:
            raise CacheCapacityError('KV cache is retained by live requests or sessions')
        allocated, replacements, jobs = [], [], []
        try:
            for pos in shared:
                src, dst = seq.block_table[pos], self._allocate_block()
                allocated.append(dst)
                valid = max(0, min(self.block_size, seq.num_cached_tokens - pos * self.block_size))
                if valid:
                    jobs.append((src, dst, valid))
                replacements.append((pos, src, dst))
            new_pages = [self._allocate_block() for _ in range(missing)]
            allocated.extend(new_pages)
            self._copy(jobs)
        except BaseException:
            self._release_ids(allocated)
            raise
        for pos, src, dst in replacements:
            seq.block_table[pos] = dst
            self._release_ids([src])
        seq.block_table.extend(new_pages)
        self._track(seq)
        self.counters['cow_pages'] += len(jobs)
        self.counters['copied_tokens'] += sum(j[2] for j in jobs)

    def stats(self):
        return dict(self.counters, live_pages=len(self.used_block_ids), free_pages=len(self.free_block_ids),
                    references=sum(b.ref_count for b in self.blocks), owners=len(self.owners),
                    shared_pages=sum(b.ref_count > 1 for b in self.blocks))

    def assert_invariants(self):
        expected = Counter(i for owner in self.owners.values() for i in owner.block_table)
        assert len(self.free_block_ids) == len(set(self.free_block_ids))
        assert set(self.free_block_ids).isdisjoint(self.used_block_ids)
        assert set(self.free_block_ids) | self.used_block_ids == set(range(len(self.blocks)))
        for block in self.blocks:
            assert block.ref_count == expected[block.block_id]
            assert (block.ref_count > 0) == (block.block_id in self.used_block_ids)
        for h, i in self.hash_to_block_id.items():
            assert self.blocks[i].hash == h
        return True

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        self.counters['peak_live_pages'] = max(self.counters['peak_live_pages'], len(self.used_block_ids))
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size
        self._track(seq)

    def deallocate(self, seq: Sequence):
        self.release(seq)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        pos = seq.num_cached_tokens // self.block_size
        return (pos < len(seq.block_table) and self.blocks[seq.block_table[pos]].ref_count == 1) or bool(self.free_block_ids)

    def may_append(self, seq: Sequence):
        pos = seq.num_cached_tokens // self.block_size
        # Ordinary decode owns its writable page exclusively: avoid constructing
        # a transaction or touching the owner ledger on this common path.
        if pos < len(seq.block_table) and self.blocks[seq.block_table[pos]].ref_count == 1:
            return
        self.ensure_writable(seq, 1)

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            assert block.ref_count == 1, "writing a shared page without COW"
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
