import argparse
import atexit
import json
import random
from pathlib import Path
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import get_context, reset_context


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--eager',action='store_true')
    ap.add_argument('--backend',choices=['force','auto'],default='force')
    args=ap.parse_args();torch.manual_seed(17);rng=random.Random(17)
    llm=LLM(args.model,enforce_eager=args.eager,max_num_seqs=48,
            max_num_batched_tokens=4096,max_model_len=4096,gpu_memory_utilization=.55,
            shared_attention_backend=args.backend)
    prefix=[rng.randrange(100,10000) for _ in range(1024)]
    llm.generate([prefix+[100]],SamplingParams(max_tokens=1,ignore_eos=True),use_tqdm=False)
    for i in range(33):
        suffix=[rng.randrange(100,10000) for _ in range((1,3,17,63,255,257)[i%6])]
        llm.scheduler.add(Sequence(prefix+suffix,SamplingParams(max_tokens=8+i%3,ignore_eos=True)))
    rows=[];runner=llm.model_runner
    while not llm.is_finished():
        seqs,prefill=llm.scheduler.schedule()
        ids,pos=runner.prepare_prefill(seqs) if prefill else runner.prepare_decode(seqs)
        ctx=get_context()
        if prefill:
            runner.run_model(ids,pos,True)
        else:
            selected=ctx.shared_enabled
            expected=runner.shared_supported and (args.backend=='force' or (runner.shared_auto_calibrated and not args.eager and len(seqs)>=32))
            assert selected == expected
            ctx.shared_enabled=False
            ref=runner.run_model(ids,pos,False).float()
            ctx.shared_enabled=selected
            got=runner.run_model(ids,pos,False).float()
            assert torch.isfinite(got).all()
            error=(ref-got).abs()
            relative_l2=(ref-got).norm()/ref.norm().clamp_min(1e-6)
            assert relative_l2.item()<0.005,relative_l2.item()
            torch.testing.assert_close(got,ref,atol=.125,rtol=.02)
            kl=(ref.softmax(-1)*(ref.log_softmax(-1)-got.log_softmax(-1))).sum(-1).clamp_min(0)
            assert kl.max().item()<0.001
            rows.append({'batch':len(seqs),'shared':selected,'max_kl':kl.max().item(),
                         'prefix':int(ctx.shared_prefix_length.item()) if selected else 0,
                         'max_abs_logit_error':error.max().item(),
                         'relative_l2':relative_l2.item(),
                         'argmax_agreement':(got.argmax(-1)==ref.argmax(-1)).float().mean().item()})
        # Identical predetermined trajectory: avoids sampling noise masking kernel errors.
        tokens=[100+(s.num_completion_tokens*17+s.seq_id)%1000 for s in seqs]
        llm.scheduler.postprocess(seqs,tokens,prefill)
        reset_context()
    assert len(rows)>=7 and any(r['batch']==33 for r in rows) and any(r['batch']<32 for r in rows)
    report={'backend':args.backend,'mode':'eager' if args.eager else 'graph','decode_steps':rows,
            'route_counts':dict(runner.attention_route_counts),
            'note':'Same model and teacher-forced history; original and shared attention compared before each step commits.'}
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    atexit.unregister(llm.exit);llm.exit()


if __name__=='__main__':main()
