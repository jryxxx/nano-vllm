"""Bounded single-common-prefix planner; immutable physical pages only."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SharedPlan:
    prefix_blocks: int = 0
    reason: str = 'off'

    @property
    def enabled(self):
        return self.prefix_blocks > 0


def select_shared_plan(seqs, block_size, mode, min_batch=32, min_prefix=1024,
                       supported=True, graph_mode=True):
    if mode == 'off':
        return SharedPlan()
    if not supported:
        return SharedPlan(reason='unsupported')
    if mode == 'auto' and not graph_mode:
        return SharedPlan(reason='eager_fallback')
    if not seqs or (mode == 'auto' and len(seqs) < min_batch):
        return SharedPlan(reason='small_batch')
    # Exclude each request's current write position / mutable last page.
    limit = min((len(s) - 1) // block_size for s in seqs)
    common = 0
    for i in range(limit):
        page = seqs[0].block_table[i]
        if page < 0 or any(s.block_table[i] != page for s in seqs[1:]):
            break
        common += 1
    if not common:
        return SharedPlan(reason='no_common_prefix')
    if mode == 'auto' and common * block_size < min_prefix:
        return SharedPlan(reason='short_prefix')
    if mode == 'auto' and max(len(s) - common * block_size for s in seqs) > common * block_size:
        return SharedPlan(reason='suffix_dominates')
    return SharedPlan(common, 'shared')
