from collections import deque
from collections import Counter
from dataclasses import dataclass
import weakref
from time import perf_counter
import math
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


class PrefixNode:
    """Exact, interned prefix identity; weak children do not retain dead caches."""
    def __init__(self, uid, parent, tokens, epoch):
        self.uid, self.parent, self.tokens = uid, parent, tokens
        self.depth = 1 if parent is None else parent.depth + 1
        self.locations = set()
        self.children = weakref.WeakSet()
        self.last_access = epoch
        self.frequency = 0.0
        self.frequency_epoch = epoch
        if parent is not None:
            parent.children.add(self)


class Block:

    def __init__(self, block_id):
        self.node = None
        self.generation = 0
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        assert self.node is None
        self.generation += 1
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, policy="fifo", half_life=32., cost_coefficients=None):
        if policy not in ('fifo', 'lru', 'leaf_lru', 'cost'):
            raise ValueError('unknown cache eviction policy')
        if not math.isfinite(half_life) or half_life <= 0:
            raise ValueError('cache half life must be finite and positive')
        self.policy, self.half_life = policy, half_life
        # Only token-work and attention-work terms: fixed launch costs are not
        # charged once per page. Without a calibration this is a work proxy.
        self.cost_coefficients = cost_coefficients or (1., 1.)
        if len(self.cost_coefficients) != 2 or any(not math.isfinite(c) or c < 0 for c in self.cost_coefficients):
            raise ValueError('invalid cache cost coefficients')
        self.epoch = 0
        self.next_uid = 0
        self.nodes = weakref.WeakValueDictionary()
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.owners = {}
        self.copy_blocks = None
        self.transfer = None
        self.counters = dict(cow_pages=0, copy_pages=0, copied_tokens=0, peak_live_pages=0, evictions=0, reused_tokens=0, admitted_tokens=0, policy_seconds=0.)

    def _node(self, parent, tokens, create=False):
        key = (0 if parent is None else parent.uid, tuple(tokens))
        node = self.nodes.get(key)
        if node is None and create:
            self.next_uid += 1
            node = PrefixNode(self.next_uid, parent, key[1], self.epoch)
            self.nodes[key] = node
        return node

    def _lookup(self, node):
        if node is None or not node.locations:
            return -1
        # Prefer a live copy: it does not consume an additional free page.
        return min(node.locations, key=lambda i: (self.blocks[i].ref_count == 0, i))

    def _attach(self, block, node):
        assert block.node is None or block.node is node
        block.node = node
        node.locations.add(block.block_id)

    def _detach(self, block):
        if block.node is not None:
            block.node.locations.remove(block.block_id)
            block.node = None

    def _touch(self, node, reuse=False):
        if node is None:
            return
        node.last_access = self.epoch
        if reuse:
            node.frequency *= 2. ** (-(self.epoch-node.frequency_epoch)/self.half_life)
            node.frequency += 1.
            node.frequency_epoch = self.epoch

    def _score(self, node):
        b = self.block_size
        a, c = self.cost_coefficients
        work = a*b/1024. + c*(b*b*(node.depth-1)+b*(b+1)/2)/1048576.
        reuse = node.frequency * 2. ** (-(self.epoch-node.frequency_epoch)/self.half_life)
        age = 2. ** (-(self.epoch-node.last_access)/self.half_life)
        return (1.+reuse)*work*age

    def _victim(self):
        if self.policy == 'fifo':
            return self.free_block_ids[0]
        start = perf_counter()
        free = [self.blocks[i] for i in self.free_block_ids]
        # Partial/unpublished pages carry no reusable prefix. Duplicate pages
        # can be reclaimed without losing the logical cached prefix.
        empty = [b for b in free if b.node is None]
        duplicates = [b for b in free if b.node is not None and len(b.node.locations)>1]
        candidates = empty or duplicates
        if not candidates:
            candidates = free
            if self.policy in ('leaf_lru', 'cost'):
                leaves = [b for b in free if not b.node.children]
                candidates = leaves or free
        def key(b):
            if b.node is None: return (0., 0, b.block_id)
            value = self._score(b.node) if self.policy == 'cost' else b.node.last_access
            return (value, -b.node.depth, b.block_id)
        result = min(candidates, key=key).block_id
        self.counters['policy_seconds'] += perf_counter()-start
        return result

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
                    table.append(dst)
                tables.append(table)
            self._copy(jobs)
        except BaseException:
            self._release_ids(allocated)
            raise
        for src, dst, valid in jobs:
            if valid == self.block_size:
                source_block = self.blocks[src]
                self.blocks[dst].update(source_block.hash, list(source_block.token_ids))
                if source_block.node is not None:
                    self._attach(self.blocks[dst], source_block.node)
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
            new_pages = []
            for _ in range(missing):
                page = self._allocate_block()
                new_pages.append(page)
                allocated.append(page)
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
                    shared_pages=sum(b.ref_count > 1 for b in self.blocks),
                    cached_free_pages=sum(b.ref_count == 0 and b.node is not None for b in self.blocks),
                    logical_nodes=len(self.nodes), policy=self.policy)

    def assert_invariants(self):
        expected = Counter(i for owner in self.owners.values() for i in owner.block_table)
        assert len(self.free_block_ids) == len(set(self.free_block_ids))
        assert set(self.free_block_ids).isdisjoint(self.used_block_ids)
        assert set(self.free_block_ids) | self.used_block_ids == set(range(len(self.blocks)))
        for block in self.blocks:
            assert block.ref_count == expected[block.block_id]
            assert (block.ref_count > 0) == (block.block_id in self.used_block_ids)
            if block.node is not None:
                assert block.block_id in block.node.locations
                assert tuple(block.token_ids) == block.node.tokens
        for node in list(self.nodes.values()):
            for i in node.locations:
                assert self.blocks[i].node is node
            assert node.parent is None or node in node.parent.children
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

    def _allocate_block(self, spill=True) -> int:
        block_id = self._victim()
        self.free_block_ids.remove(block_id)
        block = self.blocks[block_id]
        if spill and self.transfer is not None:
            try:
                self.transfer.spill(block)
            except BaseException:
                self.free_block_ids.appendleft(block_id)
                raise
        if block.node is not None:
            self.counters['evictions'] += 1
        self._detach(block)
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
        parent = None
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            parent = self._node(parent, token_ids)
            block_id = self._lookup(parent)
            if block_id == -1:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        try:
            self._allocate_sequence(seq, num_cached_blocks)
        except BaseException:
            if id(seq) in self.owners:
                self.release(seq)
            else:
                self._release_ids(seq.block_table)
            seq.block_table.clear()
            seq.num_cached_tokens = 0
            raise

    def _allocate_sequence(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        self.epoch += 1
        self.counters['admitted_tokens'] += len(seq)
        self.counters['reused_tokens'] += num_cached_blocks*self.block_size
        parent = None
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            parent = self._node(parent, token_ids)
            block_id = self._lookup(parent)
            assert block_id != -1
            block = self.blocks[block_id]
            self._touch(parent, reuse=True)
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
        # Called after successful model execution, never by scheduling probes.
        for block_id in seq.block_table:
            self._touch(self.blocks[block_id].node)
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        parent = self.blocks[seq.block_table[start - 1]].node if start > 0 else None
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            assert block.ref_count == 1, "writing a shared page without COW"
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            parent = self._node(parent, token_ids, create=True)
            self._attach(block, parent)
            self._touch(parent)
            self.hash_to_block_id[h] = block.block_id
