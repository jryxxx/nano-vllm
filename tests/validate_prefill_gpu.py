import argparse,json,time,random
from pathlib import Path
import torch
from nanovllm import LLM,SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.utils.context import reset_context

SETTINGS=[('baseline','baseline','fixed','guarded',1024),('fixed_v1','cost','fixed','guarded',1024),
          ('fixed_large','cost','fixed','guarded',4096),('model_only','cost','model','guarded',1024),
          ('joint','cost','model','predictive',1024)]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--profile',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    llm=LLM(args.model,max_num_seqs=64,max_model_len=8192,max_num_batched_tokens=8192,gpu_memory_utilization=.65,
            shared_attention_backend='auto',schedule_cost_profile=args.profile)
    reference={};report=[];runner=llm.model_runner
    for name,policy,prefill_mode,phase,chunk in SETTINGS:
        llm.config.schedule_policy='baseline';llm.config.schedule_prefill_mode=prefill_mode;llm.config.schedule_phase_mode=phase;llm.config.schedule_prefill_chunk=chunk
        sc=Scheduler(llm.config);sc.shared_capable=True;sc.block_manager.copy_blocks=runner.copy_kv_blocks;llm.scheduler=sc
        parents=[]
        for g in range(3):
            rng=random.Random(170+g);parents.append(llm.add_request([rng.randrange(100,10000) for _ in range(1151 if g==2 else 4223)],SamplingParams(max_tokens=128,ignore_eos=True)))
        while any(sc.requests[r].num_cached_tokens!=len(sc.requests[r])-1 for r in parents):llm.step()
        for rid in parents:llm.pause_request(rid)
        llm.save_session(parents[2],'prefix')
        tail=sc.requests[parents[2]].block_table[-1]
        frozen_tail=runner.kv_cache[:,:,tail,:127].clone()
        llm.config.schedule_policy=policy;clock=[0.];sc.clock=lambda:clock[0];requests=[]
        for i in range(64):requests.append(llm.fork_request(parents[i%2],SamplingParams(max_tokens=8,ignore_eos=True),next_token_id=700+i))
        for i,n in enumerate((1024,1536,4096,768)):
            rng=random.Random(300+i);requests.append(llm.add_request([rng.randrange(100,10000) for _ in range(n)],SamplingParams(max_tokens=4,ignore_eos=True)))
        requests.append(llm.fork_request(parents[0],SamplingParams(max_tokens=4,ignore_eos=True),next_token_id=899,append_token_ids=[123]*17))
        requests.append(llm.fork_request(parents[2],SamplingParams(max_tokens=4,ignore_eos=True),next_token_id=898,append_token_ids=[124]*257))
        indexes={rid:i for i,rid in enumerate(requests)};checks=[];steps=[]
        while not llm.is_finished():
            t=time.perf_counter();seqs,prefill=sc.schedule()
            ids,pos=runner.prepare_prefill(seqs) if prefill else runner.prepare_decode(seqs)
            logits=runner.run_model(ids,pos,prefill).float();torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1000
            for j,seq in enumerate(seqs):
                if prefill and seq.num_cached_tokens+seq.num_scheduled_tokens<len(seq):continue
                key=(indexes[seq.seq_id],seq.num_completion_tokens);got=logits[j]
                if name=='baseline':reference[key]=got.cpu()
                ref=reference[key].to(got.device);rel=((got-ref).norm()/ref.norm()).item()
                kl=(ref.softmax(-1)*(ref.log_softmax(-1)-got.log_softmax(-1))).sum().clamp_min(0).item()
                assert rel<.005 and kl<.001,(name,key,rel,kl)
                checks.append(dict(request=key[0],token=key[1],relative_l2=rel,max_kl=kl))
            steps.append(dict(prefill=prefill,batch=len(seqs),tokens=sum(s.num_scheduled_tokens for s in seqs),prediction=sc.prefill_prediction,elapsed_ms=elapsed))
            clock[0]+=elapsed/1000;sc.record_execution(seqs,prefill,elapsed)
            sc.postprocess(seqs,[1000+indexes[s.seq_id]+s.num_completion_tokens for s in seqs],prefill);reset_context();sc.block_manager.assert_invariants()
        assert len(checks)==536 and len({(r['request'],r['token']) for r in checks})==536
        assert torch.equal(frozen_tail,runner.kv_cache[:,:,tail,:127])
        for rid in parents:llm.cancel_request(rid)
        llm.close_session('prefix')
        assert sc.block_manager.stats()['live_pages']==0
        report.append(dict(variant=name,checks=checks,steps=steps,metrics=dict(sc.metrics),final_live_pages=0))
        print(name,'PASS',len(steps),dict(sc.metrics),flush=True)
    Path(args.output).write_text(json.dumps(dict(passed=True,cases=report),indent=2));llm.exit()

if __name__=='__main__':main()
