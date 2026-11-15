"""Matched comparison preserving previous arrivals, queue cap, and SLO criteria."""
import argparse,json
from pathlib import Path
from nanovllm import LLM
from replay_scheduler import trial

VARIANTS={
 'baseline':dict(policy='baseline',chunk=1024,prefill='fixed',phase='guarded'),
 'fixed_v1':dict(policy='cost',chunk=1024,prefill='fixed',phase='guarded'),
 'fixed_large':dict(policy='cost',chunk=4096,prefill='fixed',phase='guarded'),
 'model_only':dict(policy='cost',chunk=1024,prefill='model',phase='guarded'),
 'joint':dict(policy='cost',chunk=1024,prefill='model',phase='predictive'),
}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--profile',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--repeats',type=int,default=5)
    ap.add_argument('--scenarios',nargs='+',default=['mixed','overload','cold_only','two_hot_groups','short_hot_groups'])
    ap.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS));args=ap.parse_args()
    llm=LLM(args.model,max_num_seqs=64,max_model_len=8192,max_num_batched_tokens=8192,gpu_memory_utilization=.84,
            shared_attention_backend='auto',schedule_queue_limit=96,schedule_cost_profile=args.profile,
            schedule_decode_interval_ms=25,schedule_wait_ms=100,schedule_prefill_chunk=1024,
            schedule_service_budget_ms=70,schedule_prediction_margin=1.10,schedule_planning_margin_ms=2.)
    report=dict(config=dict(model=args.model,profile=args.profile,variants=VARIANTS,repeats=args.repeats,
             ttft_slo_ms=500,request_max_itl_slo_ms=75,queue_limit=96,service_budget_ms=70,
             prediction_margin=1.1,planning_margin_ms=2),trials=[])
    for scenario in args.scenarios:
        for repeat in range(-1,args.repeats):
            n=len(args.variants);order=args.variants[repeat%n:]+args.variants[:repeat%n]
            for variant in order:
                settings=VARIANTS[variant]
                llm.config.schedule_prefill_chunk=settings['chunk'];llm.config.schedule_prefill_mode=settings['prefill']
                llm.config.schedule_phase_mode=settings['phase']
                row=trial(llm,args,scenario,settings['policy'],repeat);row['variant']=variant
                if repeat>=0:report['trials'].append(row);Path(args.output).write_text(json.dumps(report,indent=2))
                print(json.dumps({k:v for k,v in row.items() if k not in ('records','steps')}),flush=True)
    llm.exit()

if __name__=='__main__':main()
