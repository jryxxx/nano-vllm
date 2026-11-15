"""Bounded asynchronous cache spill/restore for single-GPU inference.

D2H sources are dedicated GPU staging slots. Main-stream snapshot copies precede
reuse of evicted KV slots; staging/CPU slots remain pinned until completion.
"""
from dataclasses import dataclass
from collections import Counter
import torch


@dataclass(eq=False)
class RestoreTicket:
    block_table: list
    entries: list
    event: object
    request_id: int


class TransferManager:
    def __init__(self, bm, kv, config, cost):
        self.bm, self.kv, self.config, self.cost = bm, kv, config, cost
        shape=(kv.shape[0],kv.shape[1],*kv.shape[3:])
        self.cpu=torch.empty((config.cpu_cache_pages,*shape),dtype=kv.dtype,device='cpu',pin_memory=True)
        self.staging=torch.empty((min(config.cpu_cache_staging,config.cpu_cache_pages),*shape),dtype=kv.dtype,device=kv.device)
        self.stream=torch.cuda.Stream(device=kv.device)
        self.entries={};self.stage_events={};self.tickets={};self.stats=Counter()
        self.page_bytes=self.cpu[0].numel()*self.cpu.element_size()
        # Measure the actual non-contiguous destination used by restore, not
        # a faster contiguous staging copy. The scratch slot is never published.
        scratch=bm._allocate_block(spill=False)
        timings={'h2d':[], 'd2h':[]}
        try:
            for direction in timings:
                for _ in range(4):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(self.stream):
                        start.record()
                        if direction=='h2d':self._copy_in(scratch,0)
                        else:self._copy_out(0,0)
                        end.record()
                    end.synchronize();timings[direction].append(start.elapsed_time(end))
        finally:
            self.stream.synchronize()
            bm._release_ids([scratch])
        self.page_ms=sorted(timings['h2d'][1:])[1]
        self.page_out_ms=sorted(timings['d2h'][1:])[1]
        bm.transfer=self

    def poll(self):
        for stage,event in list(self.stage_events.items()):
            if event.query():del self.stage_events[stage]
        for entry in self.entries.values():
            if entry['state']=='COPYING_OUT' and entry['event'].query():entry['state']='CPU_READY'
        for rid,ticket in list(self.tickets.items()):
            if not ticket.event.query():continue
            for block_id,entry in zip(ticket.block_table,ticket.entries):
                block=self.bm.blocks[block_id]
                block.update(entry['hash'],list(entry['node'].tokens))
                self.bm._attach(block,entry['node'])
                self.bm.hash_to_block_id[block.hash]=block_id
                entry['state']='CPU_READY'
            self.bm.release(ticket)
            del self.tickets[rid]
            self.stats['restored_pages']+=len(ticket.block_table)

    def _copy_out(self, slot, stage):
        self.cpu[slot].copy_(self.staging[stage],non_blocking=True)

    def _copy_in(self, block_id, slot):
        self.kv[:,:,block_id].copy_(self.cpu[slot],non_blocking=True)

    def spill(self, block):
        self.poll()
        node=block.node
        if node is None or len(node.locations)>1 or node.uid in self.entries:return
        stage=next((i for i in range(len(self.staging)) if i not in self.stage_events),None)
        if stage is None:
            self.stats['spill_queue_full']+=1;return
        occupied={e['slot'] for e in self.entries.values()}
        slot=next((i for i in range(len(self.cpu)) if i not in occupied),None)
        if slot is None:
            ready=[e for e in self.entries.values() if e['state']=='CPU_READY']
            if not ready:
                self.stats['cpu_pool_busy']+=1;return
            victim=min(ready,key=lambda e:e['node'].last_access)
            slot=victim['slot'];del self.entries[victim['node'].uid]
            self.stats['cpu_evictions']+=1
        ready,done=torch.cuda.Event(),torch.cuda.Event()
        # Source may be reused later on the same main stream, after this snapshot.
        self.staging[stage].copy_(self.kv[:,:,block.block_id],non_blocking=True)
        ready.record(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(ready)
                self._copy_out(slot,stage)
                done.record()
        except BaseException:
            ready.synchronize()
            self.stream.synchronize()
            self.stats['spill_failures']+=1
            raise
        self.stage_events[stage]=done
        self.entries[node.uid]=dict(node=node,slot=slot,hash=block.hash,state='COPYING_OUT',event=done)
        self.stats['spilled_pages']+=1;self.stats['d2h_bytes']+=self.page_bytes

    def prepare(self, seq):
        self.poll()
        if seq.seq_id in self.tickets:return False
        if seq.block_table:return True
        parent=None;chain=[];missing=[];gpu_prefix=0
        for i in range(seq.num_blocks-1):
            node=self.bm._node(parent,seq.block(i))
            if node is None:break
            block=self.bm._lookup(node)
            if block<0:
                entry=self.entries.get(node.uid)
                if entry is None or entry['state']!='CPU_READY':break
                missing.append(entry)
            elif not missing:gpu_prefix+=1
            chain.append(node);parent=node
        if not missing:return True
        # Reserve enough free pages for both restored prefix and uncached suffix.
        live=sum(self.bm._lookup(n) in self.bm.used_block_ids for n in chain)
        if len(self.bm.free_block_ids)<seq.num_blocks-live:
            self.stats['restore_capacity_skip']+=1;return True
        predicted=len(missing)*self.page_ms+len(self.stage_events)*self.page_out_ms
        if predicted>self.config.cpu_cache_restore_budget_ms:
            self.stats['restore_budget_skip']+=1;return True
        if self.config.cpu_cache_restore=='auto':
            b=self.bm.block_size
            before=self.cost.predict_prefill([len(seq)-gpu_prefix*b],[gpu_prefix*b])
            after=self.cost.predict_prefill([len(seq)-len(chain)*b],[len(chain)*b])
            if before is None or after is None or predicted>=max(0.,before-after):
                self.stats['restore_cost_skip']+=1;return True
        # Temporarily protect existing GPU prefix copies while allocating destinations.
        held=[]
        for node in chain:
            block_id=self.bm._lookup(node)
            if block_id>=0:
                block=self.bm.blocks[block_id]
                if block.ref_count==0:
                    self.bm.free_block_ids.remove(block_id);self.bm.used_block_ids.add(block_id)
                block.ref_count+=1;held.append(block_id)
        allocated=[]
        try:
            for entry in missing:
                entry['state']='COPYING_IN'
                # No recursive spilling while reserving a restore transaction.
                block_id=self.bm._allocate_block(spill=False);allocated.append(block_id)
            event=torch.cuda.Event()
            main_stream=torch.cuda.current_stream(self.kv.device)
            with torch.cuda.stream(self.stream):
                self.stream.wait_stream(main_stream)
                for block_id,entry in zip(allocated,missing):
                    self._copy_in(block_id,entry['slot'])
                event.record()
            ticket=RestoreTicket(allocated,missing,event,seq.seq_id)
            self.bm._track(ticket);self.tickets[seq.seq_id]=ticket
            self.stats['h2d_bytes']+=len(missing)*self.page_bytes
        except BaseException:
            # A partial asynchronous transfer must complete before destination reuse.
            self.stream.synchronize()
            self.stats['restore_failures']+=1
            self.bm._release_ids(allocated)
            for entry in missing:entry['state']='CPU_READY'
            raise
        finally:self.bm._release_ids(held)
        return False

    def wait(self):
        if self.tickets:
            next(iter(self.tickets.values())).event.synchronize();self.poll()

    def close(self):
        self.stream.synchronize();self.poll()
        assert not self.tickets
        self.entries.clear();self.stage_events.clear();self.bm.transfer=None

    def metrics(self):
        self.poll()
        return dict(self.stats,cpu_pages=len(self.entries),pinned_bytes=self.cpu.numel()*self.cpu.element_size(),
                    staging_bytes=self.staging.numel()*self.staging.element_size(),page_h2d_ms=self.page_ms,page_d2h_ms=self.page_out_ms,
                    pending_restores=len(self.tickets))
