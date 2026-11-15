import os
"""Closed-loop cache-pressure replay; identical ordered requests for every policy."""
import argparse,json,time,random,importlib.util
from pathlib import Path
import numpy as np
import torch
from nanovllm import LLM,SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context

POLICIES=['fifo','lru','leaf_lru','cost']

def trace(tokenizer,scenario,n=48):
    def document(family,length):
        text=f'Document identity: {family}.\nTechnical review: memory management, scheduling and reproducible experiments.\n'
        ids=tokenizer.encode(text+('Each evaluation records its inputs, outputs, resource usage and elapsed time. '*1000))
        return (ids*((length//len(ids))+1))[:length]
    requests=[]
    for i in range(n):
        if scenario=='pollution':
            family,length=(i%2,2048) if i%4!=3 else (100+i,3072)
        elif scenario=='switch':
            family,length=((i%2)+(0 if i<n//2 else 10),2048) if i%4!=3 else (100+i,3072)
        elif scenario=='stream':family,length=100+i,(1024,2048,3072)[i%3]
        elif scenario=='mixed_cost':
            family,length=((i%3),(1024,2048,3072)[i%3]) if i%5!=4 else (100+i,2048)
        else:raise ValueError(scenario)
        requests.append(document(family,length)+[100+i])
    return requests

def manager(llm,policy,pages):
    assert llm.is_finished()
    co=llm.scheduler.cost.prefill['coefficients']
    if policy=='previous_fifo':
        spec=importlib.util.spec_from_file_location('old_block_manager',str(Path(__file__).resolve().parents[1] / 'tests/fixtures/pre_cache_policy_block_manager.py'))
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        bm=old.BlockManager(pages,256)
    else:
        bm=BlockManager(pages,256,policy,32.,(co[2],co[3]))
    bm.copy_blocks=llm.model_runner.copy_kv_blocks
    llm.scheduler.block_manager=bm
    return bm

def validate(llm):
    runner=llm.model_runner;refs={};previous={};rows=[]
    requests=trace(llm.tokenizer,'switch',32)
    for policy in ['oracle','previous_fifo']+POLICIES:
        bm=manager(llm,'fifo' if policy=='oracle' else policy,32)
        checks=[]
        for i,prompt in enumerate(requests):
            if policy=='oracle':bm=manager(llm,'fifo',32)
            llm.add_request(prompt,SamplingParams(max_tokens=3,ignore_eos=True))
            while not llm.is_finished():
                seqs,prefill=llm.scheduler.schedule()
                ids,pos=runner.prepare_prefill(seqs) if prefill else runner.prepare_decode(seqs)
                logits=runner.run_model(ids,pos,prefill).float()
                for j,seq in enumerate(seqs):
                    key=(i,seq.num_completion_tokens);got=logits[j]
                    if policy=='oracle':refs[key]=got.cpu()
                    ref=refs[key].to(got.device)
                    rel=((got-ref).norm()/ref.norm()).item()
                    kl=(ref.softmax(-1)*(ref.log_softmax(-1)-got.log_softmax(-1))).sum().clamp_min(0).item()
                    if policy=='previous_fifo':previous[key]=got.cpu()
                    previous_error=(got.cpu()-previous[key]).abs().max().item() if policy=='fifo' else None
                    previous_rel=((got.cpu()-previous[key]).norm()/previous[key].norm()).item() if policy=='fifo' else None
                    if policy=='fifo':assert previous_rel<.005,(key,previous_rel)
                    checks.append(dict(request=i,token=key[1],relative_l2=rel,kl=kl,previous_fifo_max_abs=previous_error,previous_fifo_relative_l2=previous_rel))
                llm.scheduler.postprocess(seqs,[500+i+s.num_completion_tokens for s in seqs],prefill)
                reset_context();bm.assert_invariants()
        rows.append(dict(policy=policy,checks=checks,stats=bm.stats()))
        print('VALIDATED',policy,flush=True)
    return dict(fifo_relative_l2_regression_passed=True,full_recompute_gate_passed=all(c['relative_l2']<.005 and c['kl']<.001 for r in rows for c in r['checks']),cases=rows)

def bench(llm,repeats,capacities):
    rows=[]
    for scenario in ('pollution','switch','stream','mixed_cost'):
        requests=trace(llm.tokenizer,scenario)
        for pages in capacities:
            for repeat in range(-1,repeats):
                order=POLICIES[repeat%4:]+POLICIES[:repeat%4]
                for policy in order:
                    bm=manager(llm,policy,pages);raw=[]
                    torch.cuda.synchronize();start=time.perf_counter()
                    for i,prompt in enumerate(requests):
                        before=bm.counters['reused_tokens'];t=time.perf_counter()
                        llm.add_request(prompt,SamplingParams(max_tokens=4,ignore_eos=True,temperature=.6))
                        first=None
                        while not llm.is_finished():
                            llm.step()
                            if first is None:first=time.perf_counter()
                        raw.append(dict(request=i,input_tokens=len(prompt),cached_tokens=bm.counters['reused_tokens']-before,
                                        ttft_ms=(first-t)*1000,latency_ms=(time.perf_counter()-t)*1000))
                    torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    bm.assert_invariants();assert not bm.used_block_ids
                    row=dict(scenario=scenario,pages=pages,repeat=repeat,policy=policy,elapsed_s=elapsed,
                        output_tokens_per_s=4*len(requests)/elapsed,ttft_p50_ms=float(np.percentile([r['ttft_ms'] for r in raw],50)),
                        ttft_p95_ms=float(np.percentile([r['ttft_ms'] for r in raw],95)),
                        stats=bm.stats(),requests=raw)
                    if repeat<0:
                        print("WARMUP",scenario,pages,policy,flush=True)
                        continue
                    rows.append(row)
                    print('TRIAL',scenario,pages,repeat,policy,round(row['output_tokens_per_s'],2),bm.counters['reused_tokens'],flush=True)
    return dict(trials=rows,note='Full per-policy trace warmup excluded; sequential closed-loop, fixed 4 output tokens, same traces; logical pool cap within unchanged physical GPU allocation. Not concurrent SLO throughput.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['validate','bench'],required=True);ap.add_argument('--output',required=True);ap.add_argument('--repeats',type=int,default=3);ap.add_argument('--pages',type=int,nargs='+',default=[32,64]);args=ap.parse_args()
    torch.manual_seed(20261114)
    llm=LLM(os.environ.get('NANO_MODEL', '/root/autodl-tmp/shared-context/models/Qwen3-0.6B-fp16'),max_num_seqs=8,max_model_len=4096,max_num_batched_tokens=4096,
            gpu_memory_utilization=.45,schedule_cost_profile='results/finalization/cost-profile.json')
    for length in (1025,2049,3073):llm.generate([[100]*length],SamplingParams(max_tokens=4,ignore_eos=True),use_tqdm=False)
    result=validate(llm) if args.mode=='validate' else bench(llm,args.repeats,args.pages)
    Path(args.output).write_text(json.dumps(result,indent=2));llm.exit()
    if args.mode=="validate":assert result["full_recompute_gate_passed"], "strict numerical gate failed"
if __name__=='__main__':main()
