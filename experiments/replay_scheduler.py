"""Open-loop replay with preset arrivals, natural-language formatted inputs and explicit admission loss."""
import argparse,json,time,statistics,random
from pathlib import Path
import numpy as np
import torch
from nanovllm import LLM,SamplingParams
from nanovllm.engine.scheduler import Scheduler,RequestQueueFull

POLICIES=['baseline','fcfs','lpm','cost']
SCENARIOS=['two_hot_groups','short_hot_groups','mixed','overload','cold_only']
SLO_TTFT_MS=500.;SLO_MAX_ITL_MS=75.


def percentiles(x):
    return {str(p):float(np.percentile(x,p)) if x else None for p in (50,95,99,100)}


def text_tokens(tokenizer,label,n):
    text=(f'会话资料：{label}。请根据资料回答问题。\n'+
          (f'{label}的设计文档说明：系统需要记录请求到达时间、缓存所有权和资源使用。'
           '处理共享上下文时，完整历史可以复用，独立分支必须保持各自的输出。'
           '评估应同时关注吞吐、首个输出的等待时间以及不同请求之间的公平性。\n')*160)
    ids=tokenizer.encode(text)
    return (ids*((n+len(ids)-1)//len(ids)))[:n]


def trial(llm,args,scenario,policy,repeat):
    config=llm.config;config.schedule_policy=policy
    sc=Scheduler(config);sc.shared_capable=llm.model_runner.shared_auto_calibrated
    sc.block_manager.copy_blocks=llm.model_runner.copy_kv_blocks;llm.scheduler=sc
    torch.manual_seed(200+max(repeat,0));parents=[]
    if scenario!='cold_only':
        # Session preparation is outside online client latency; no future user
        # request is injected early. Subsequent fork arrivals are explicitly dated.
        config.schedule_policy='baseline'
        parents=[llm.add_request(text_tokens(llm.tokenizer,f'主题{g}',1151 if scenario=='short_hot_groups' else 4223),SamplingParams(max_tokens=128,ignore_eos=True)) for g in range(2)]
        while any(sc.requests[rid].num_cached_tokens != len(sc.requests[rid])-1 for rid in parents):
            llm.step()
        for rid in parents:llm.pause_request(rid)
        config.schedule_policy=policy
    events=[]
    if parents:
        events += [dict(at=0.,kind='hot',parent=i%2,index=i,output=64) for i in range(64)]
    count,period=(16,.02) if scenario=='mixed' else ((96,.005) if scenario=='overload' else ((32,.01) if scenario=='cold_only' else (0,0)))
    for i in range(count):
        events.append(dict(at=i*period,kind='cold',index=i,output=8 if scenario!='cold_only' else 32,
                           tokens=text_tokens(llm.tokenizer,f'独立任务编号{i}',4096 if scenario!='cold_only' else 1024)))
    events.sort(key=lambda e:e['at'])
    records=[];by_id={};step_rows=[];preemptions=[];original_preempt=sc.preempt
    def preempt(seq):preemptions.append(seq.seq_id);return original_preempt(seq)
    sc.preempt=preempt;original_post=sc.postprocess
    def post(seqs,ids,prefill):
        before={s.seq_id:s.num_completion_tokens for s in seqs}
        original_post(seqs,ids,prefill);now=time.perf_counter()
        for s in seqs:
            if s.seq_id in by_id and s.num_completion_tokens>before[s.seq_id]:by_id[s.seq_id]['times'].append(now)
    sc.postprocess=post;llm.model_runner.attention_route_counts.clear();sc.metrics.clear()
    start=time.perf_counter();cursor=0;peak_queue=0
    while cursor<len(events) or not llm.is_finished():
        now=time.perf_counter()
        while cursor<len(events) and start+events[cursor]['at']<=now:
            event=events[cursor];cursor+=1
            record=dict(kind=event['kind'],arrival=start+event['at'],times=[],rejected=False,expected=event['output'])
            records.append(record)
            try:
                sp=SamplingParams(max_tokens=event['output'],ignore_eos=True,temperature=.6)
                if event['kind']=='hot':rid=llm.fork_request(parents[event['parent']],sp,next_token_id=700+event['index'])
                else:rid=llm.add_request(event['tokens'],sp)
                by_id[rid]=record
            except RequestQueueFull:record['rejected']=True
            now=time.perf_counter()
        peak_queue=max(peak_queue,len(sc.waiting))
        if not llm.is_finished():
            t=time.perf_counter();done,work=llm.step();elapsed=(time.perf_counter()-t)*1000
            step_rows.append(dict(ms=elapsed,work=work,waiting=len(sc.waiting),running=len(sc.running)))
        elif cursor<len(events):time.sleep(min(.001,max(0,start+events[cursor]['at']-time.perf_counter())))
        if time.perf_counter()-start>90:raise RuntimeError('replay failed to drain within 90 seconds')
    elapsed=time.perf_counter()-start;results=[]
    for r in records:
        times=r['times'];intervals=[(b-a)*1000 for a,b in zip(times,times[1:])]
        ttft=(times[0]-r['arrival'])*1000 if times else None
        finished=len(times)==r['expected'];max_itl=max(intervals,default=0.)
        results.append(dict(kind=r['kind'],arrival_s=r['arrival']-start,rejected=r['rejected'],finished=finished,
                            ttft_ms=ttft,max_itl_ms=max_itl if times else None,
                            completion_ms=(times[-1]-r['arrival'])*1000 if times else None,
                            good=finished and ttft<=SLO_TTFT_MS and max_itl<=SLO_MAX_ITL_MS,
                            token_times_s=[t-start for t in times]))
    for rid in parents:llm.cancel_request(rid)
    sc.block_manager.assert_invariants();assert sc.block_manager.stats()['live_pages']==0
    assert all(r['finished'] or r['rejected'] for r in results)
    cohorts={}
    for kind in ('hot','cold'):
        group=[r for r in results if r['kind']==kind]
        if not group:continue
        accepted=[r for r in group if not r['rejected']]
        cohorts[kind]=dict(offered=len(group),rejected=sum(r['rejected'] for r in group),good=sum(r['good'] for r in group),
                           ttft_ms=percentiles([r['ttft_ms'] for r in accepted]),
                           request_max_itl_ms=percentiles([r['max_itl_ms'] for r in accepted]))
    return dict(scenario=scenario,policy=policy,repeat=repeat,elapsed_s=elapsed,offered=len(results),
                accepted=sum(not r['rejected'] for r in results),rejected=sum(r['rejected'] for r in results),unfinished=0,
                good=sum(r['good'] for r in results),goodput=sum(r['good'] for r in results)/elapsed,
                output_tokens_per_s=sum(len(r['times']) for r in records)/elapsed,
                cohorts=cohorts,peak_waiting=peak_queue,preemptions=len(preemptions),
                routes=dict(llm.model_runner.attention_route_counts),scheduler_metrics=dict(sc.metrics),
                peak_live_pages=sc.block_manager.stats()['peak_live_pages'],records=results,steps=step_rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--profile',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--repeats',type=int,default=4)
    ap.add_argument('--scenarios',nargs='+',default=SCENARIOS)
    ap.add_argument('--policies',nargs='+',choices=POLICIES,default=POLICIES)
    ap.add_argument('--backend',choices=['auto','off'],default='auto');args=ap.parse_args()
    llm=LLM(args.model,max_num_seqs=64,max_model_len=8192,max_num_batched_tokens=8192,gpu_memory_utilization=.84,
            shared_attention_backend='auto',schedule_queue_limit=96,schedule_cost_profile=args.profile,
            schedule_decode_interval_ms=25,schedule_wait_ms=100,schedule_prefill_chunk=1024)
    llm.config.shared_attention_backend=args.backend
    report=dict(config=dict(model=args.model,profile=args.profile,attention_backend=args.backend,ttft_slo_ms=SLO_TTFT_MS,request_max_itl_slo_ms=SLO_MAX_ITL_MS,
                            queue_limit=96,decode_interval_ms=25,wait_ms=100,repeats=args.repeats,source='natural-language formatted synthetic documents'),trials=[])
    for scenario in args.scenarios:
        for repeat in range(-1,args.repeats):
            n=len(args.policies)
            order=args.policies[repeat%n:]+args.policies[:repeat%n]
            for policy in order:
                row=trial(llm,args,scenario,policy,repeat)
                if repeat>=0:report['trials'].append(row);Path(args.output).write_text(json.dumps(report,indent=2))
                print(json.dumps({k:v for k,v in row.items() if k not in ('records','steps')}),flush=True)
    llm.exit()

if __name__=='__main__':main()
