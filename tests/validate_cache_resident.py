import os
"""Bitwise resident KV protection under eviction pressure and retained sessions."""
import json
from pathlib import Path
import torch
from nanovllm import LLM,SamplingParams
from experiments.bench_cache import manager,trace,POLICIES

llm=LLM(os.environ.get('NANO_MODEL', '/root/autodl-tmp/shared-context/models/Qwen3-0.6B-fp16'),max_num_seqs=8,max_model_len=4096,
        max_num_batched_tokens=4096,gpu_memory_utilization=.45,schedule_cost_profile='results/finalization/cost-profile.json')
rows=[]
for policy in POLICIES:
    bm=manager(llm,policy,20);runner=llm.model_runner
    rid=llm.add_request([321]*511,SamplingParams(max_tokens=4,ignore_eos=True))
    llm.step();llm.pause_request(rid);llm.save_session(rid,'held');llm.cancel_request(rid)
    snap=llm.scheduler.sessions['held']
    frozen=[(p,min(256,511-i*256),runner.kv_cache[:,:,p,:min(256,511-i*256)].clone()) for i,p in enumerate(snap.block_table)]
    shadow={};checks=0
    def audit():
        global checks
        resident={b.block_id:b for b in bm.blocks if b.node is not None}
        for key in list(shadow):
            i,g=key
            if i not in resident or resident[i].generation!=g:del shadow[key]
        for i,b in resident.items():
            key=(i,b.generation);data=runner.kv_cache[:,:,i]
            if key in shadow:
                assert torch.equal(data,shadow[key]),(policy,i,b.generation)
                checks+=1
            else:shadow[key]=data.clone()
        for page,valid,old in frozen:assert torch.equal(old,runner.kv_cache[:,:,page,:valid])
        bm.assert_invariants()
    for prompt in trace(llm.tokenizer,'switch',24):
        llm.add_request(prompt,SamplingParams(max_tokens=3,ignore_eos=True))
        while not llm.is_finished():audit();llm.step();audit()
    llm.close_session('held');bm.assert_invariants();assert bm.stats()['live_pages']==0
    assert bm.stats()['evictions']>0
    rows.append(dict(policy=policy,bitwise_page_checks=checks,stats=bm.stats()))
    print(policy,'PASS',checks,flush=True)
Path('results/cache/resident-validation.json').write_text(json.dumps(dict(passed=True,cases=rows),indent=2))
llm.exit()
