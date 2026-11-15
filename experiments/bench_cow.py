"""Matched bulk forks: include deferred COW in fork-to-first-token and total latency."""
import argparse
import json
import random
import time
from pathlib import Path
import torch
from nanovllm import LLM, SamplingParams


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True)
    ap.add_argument('--output',required=True); ap.add_argument('--repeats',type=int,default=3)
    args=ap.parse_args()
    llm=LLM(args.model,max_num_seqs=32,max_model_len=8192,max_num_batched_tokens=8192,
            gpu_memory_utilization=.68,shared_attention_backend='off')
    bm=llm.scheduler.block_manager; runner=llm.model_runner; rng=random.Random(2026)
    bytes_per_token=2*runner.kv_cache.shape[1]*runner.kv_cache.shape[-2]*runner.kv_cache.shape[-1]*runner.kv_cache.element_size()
    report=dict(config=dict(dtype=str(runner.kv_cache.dtype),gpu=torch.cuda.get_device_name(),
                            pool_pages=len(bm.blocks),bytes_per_token=bytes_per_token,block_size=bm.block_size,
                            backend='off',outputs=32,repeats=args.repeats),trials=[])
    for cached in (256,1151,4223):
        rid=llm.add_request([rng.randrange(100,10000) for _ in range(cached)],SamplingParams(max_tokens=64,ignore_eos=True))
        llm.step(); llm.pause_request(rid)
        for branches in (1,8,16):
            for repeat in range(-1,args.repeats):
                modes=['copy','cow'] if repeat%2 else ['cow','copy']
                for mode in modes:
                    before=bm.stats(); bm.counters['peak_live_pages']=before['live_pages']
                    torch.manual_seed(400+max(repeat,0)); torch.cuda.synchronize()
                    t0=time.perf_counter()
                    children=llm.fork_requests(rid,branches,SamplingParams(max_tokens=32,temperature=.6,ignore_eos=True),
                                               next_token_ids=list(range(700,700+branches)),cache_mode=mode)
                    fork=time.perf_counter()-t0
                    live_fork=bm.stats()['live_pages']
                    llm.step(); first=time.perf_counter()-t0
                    live_first=bm.stats()['live_pages']; outputs={}
                    while not llm.is_finished():
                        done,_=llm.step(); outputs.update(done)
                    total=time.perf_counter()-t0
                    assert len(outputs)==branches and all(len(x)==32 for x in outputs.values())
                    after=bm.stats(); bm.assert_invariants()
                    assert after['live_pages']==before['live_pages']
                    row=dict(cached_tokens=cached,branches=branches,repeat=repeat,mode=mode,
                             fork_ms=fork*1000,fork_to_first_ms=first*1000,total_ms=total*1000,
                             tokens_per_s=branches*32/total,live_after_fork=live_fork,live_after_first=live_first,
                             peak_live_pages=after['peak_live_pages'],parent_pages=before['live_pages'],
                             copied_tokens=after['copied_tokens']-before['copied_tokens'],
                             copy_bytes=(after['copied_tokens']-before['copied_tokens'])*bytes_per_token,
                             cow_pages=after['cow_pages']-before['cow_pages'],copy_pages=after['copy_pages']-before['copy_pages'])
                    if repeat>=0:
                        report['trials'].append(row)
                        Path(args.output).write_text(json.dumps(report,indent=2))
                    print(json.dumps(row),flush=True)
        llm.cancel_request(rid); bm.assert_invariants(); assert bm.stats()['live_pages']==0
    report['final_stats']=bm.stats(); Path(args.output).write_text(json.dumps(report,indent=2))
    llm.exit()


if __name__=='__main__': main()
