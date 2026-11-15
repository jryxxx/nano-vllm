"""Teacher-forced six-arm whole-model equivalence and route-transition checks."""
import argparse
import json
import random
from pathlib import Path
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.utils.context import get_context,reset_context

VARIANTS=[('copy','off'),('copy','auto'),('cow','off'),('cow','auto'),('eager_tail','off'),('eager_tail','auto')]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    llm=LLM(args.model,max_num_seqs=48,max_model_len=8192,max_num_batched_tokens=16384,
            gpu_memory_utilization=.84,shared_attention_backend='auto')
    runner=llm.model_runner; scheduler=llm.scheduler; report=[]
    for scenario,p,n in [('long',4223,32),('shrinking',1151,33),('continuation',1151,32)]:
        baseline={}; prompt=[random.Random(77+i).randrange(100,10000) for i in range(p)]
        for mode,backend in VARIANTS:
            llm.config.shared_attention_backend=backend
            bm=BlockManager(len(scheduler.block_manager.blocks),256); bm.copy_blocks=runner.copy_kv_blocks; scheduler.block_manager=bm
            rid=llm.add_request(prompt,SamplingParams(max_tokens=16,ignore_eos=True));llm.step();llm.pause_request(rid)
            children=[]
            for i in range(n):
                count=16+16*(i%3) if scenario=='shrinking' else (64 if scenario=='long' else 4)
                extension=[900+i]*[0,1,17,257][i%4] if scenario=='continuation' else []
                child=llm.fork_request(rid,SamplingParams(max_tokens=count,ignore_eos=True),next_token_id=700+i,
                                       append_token_ids=extension,cache_mode='copy' if mode=='copy' else 'cow')
                children.append(child)
                if mode=='eager_tail':bm.ensure_writable(scheduler.requests[child],1)
            index={rid:i for i,rid in enumerate(children)}; step=0; checks=[]; runner.attention_route_counts.clear()
            while not llm.is_finished():
                seqs,prefill=scheduler.schedule()
                ids,pos=runner.prepare_prefill(seqs) if prefill else runner.prepare_decode(seqs)
                selected=get_context().shared_enabled
                logits=runner.run_model(ids,pos,prefill).float()
                order=[index[s.seq_id] for s in seqs]
                if mode=='copy' and backend=='off':baseline[step]=(order,logits.cpu())
                reference_order,reference=baseline[step]; assert order==reference_order
                ref=reference.to(logits.device); diff=logits-ref
                rel=(diff.norm()/ref.norm()).item()
                kl=(ref.softmax(-1)*(ref.log_softmax(-1)-logits.log_softmax(-1))).sum(-1).clamp_min(0).max().item()
                if backend=='off' or mode=='copy':
                    assert torch.equal(logits,ref), (scenario,mode,backend,step,diff.abs().max().item())
                else:
                    assert rel<.005 and kl<.001,(rel,kl)
                    torch.testing.assert_close(logits,ref,atol=.125,rtol=.02)
                checks.append(dict(step=step,batch=len(seqs),prefill=prefill,shared=selected,relative_l2=rel,
                                   max_kl=kl,max_abs=diff.abs().max().item(),
                                   argmax_matches=int((ref.argmax(-1)==logits.argmax(-1)).sum()),rows=len(seqs)))
                tokens=[1000+index[s.seq_id]*7+s.num_completion_tokens for s in seqs]
                scheduler.postprocess(seqs,tokens,prefill); reset_context(); bm.assert_invariants(); step+=1
            llm.cancel_request(rid);bm.assert_invariants();assert bm.stats()['live_pages']==0
            routes=dict(runner.attention_route_counts)
            assert (routes.get('shared',0)>0)==(mode!='copy' and backend=='auto')
            if scenario=='shrinking' and backend=='auto':assert routes.get('small_batch',0)>0
            report.append(dict(scenario=scenario,mode=mode,backend=backend,checks=checks,routes=routes,final_live_pages=0))
            print(scenario,mode,backend,'PASS',routes,flush=True)
    Path(args.output).write_text(json.dumps(dict(passed=True,cases=report),indent=2));llm.exit()


if __name__=='__main__':main()
