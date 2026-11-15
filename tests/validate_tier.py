import os
import json,torch
from pathlib import Path
from nanovllm import LLM,SamplingParams
from experiments.bench_cache import trace
from nanovllm.engine.sequence import Sequence

llm=LLM(os.environ.get('NANO_MODEL', '/root/autodl-tmp/shared-context/models/Qwen3-0.6B-fp16'),max_num_seqs=8,max_model_len=4096,
 max_num_batched_tokens=4096,gpu_memory_utilization=.5,max_kvcache_blocks=20,cpu_cache_pages=64,
 cpu_cache_staging=4,cpu_cache_restore='always',cpu_cache_restore_budget_ms=100,
 schedule_cost_profile='results/finalization/cost-profile.json',cache_policy='lru')
bm=llm.scheduler.block_manager;tm=llm.transfer_manager;checks=0;raw=[]
for prompt in trace(llm.tokenizer,'switch',48):
 rid=llm.add_request(prompt,SamplingParams(max_tokens=3,ignore_eos=True))
 while not llm.is_finished():
  llm.step();tm.poll();bm.assert_invariants()
  # Every ready CPU copy must equal each still-resident exact logical identity.
  for entry in tm.entries.values():
   if entry['state']=='CPU_READY':
    for page in entry['node'].locations:
     assert torch.equal(tm.cpu[entry['slot']],tm.kv[:,:,page].cpu())
     checks+=1
 raw.append(tm.metrics())
assert tm.stats['spilled_pages']>0 and tm.stats['restored_pages']>0,tm.metrics()
tm.stream.synchronize();tm.poll();bm.assert_invariants();assert not bm.used_block_ids
# Failure after an actual asynchronous copy must release destination reservations.
ready=next(e for e in tm.entries.values() if e['node'].depth==1 and not e['node'].locations)
probe=Sequence(list(ready['node'].tokens)+[99],SamplingParams(max_tokens=2,ignore_eos=True))
original=tm._copy_in
calls=[0]
def fail_in(page,slot):
 original(page,slot);calls[0]+=1;raise RuntimeError('injected H2D failure')
tm._copy_in=fail_in
try:tm.prepare(probe);assert False
except RuntimeError as error:assert 'injected' in str(error)
finally:tm._copy_in=original
assert calls[0]==1 and not tm.tickets and not bm.used_block_ids
bm.assert_invariants()
# Cancellation retains transfer ownership until its event completes.
assert not tm.prepare(probe)
llm.scheduler.add(probe);assert llm.cancel_request(probe.seq_id)
bm.assert_invariants();tm.stream.synchronize();tm.poll()
bm.assert_invariants();assert not bm.used_block_ids
# Inject D2H failure through the allocator, testing whole-allocation rollback.
block=next(b for b in bm.blocks if b.node is not None and len(b.node.locations)==1)
old=tm.entries.pop(block.node.uid,None)
original_out=tm._copy_out;victim=bm._victim
calls=[0]
def fail_out(slot,stage):
 original_out(slot,stage);calls[0]+=1;raise RuntimeError('injected D2H failure')
tm._copy_out=fail_out;bm._victim=lambda:block.block_id
probe=Sequence([987]*257)
try:bm.allocate(probe,0);assert False
except RuntimeError as error:assert 'injected' in str(error)
finally:tm._copy_out=original_out;bm._victim=victim
assert calls[0]==1 and not probe.block_table
bm.assert_invariants();assert not bm.used_block_ids
result=dict(failure_rollback=True,cancel_during_restore=True,passed=True,bitwise_checks=checks,metrics=tm.metrics(),requests=raw)
Path('results/finalization/tier-validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='requests'}));llm.exit()
