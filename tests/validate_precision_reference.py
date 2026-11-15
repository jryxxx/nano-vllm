import os
"""Compare residual precision against an independent FP32 Hugging Face implementation."""
import json,random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from nanovllm import LLM,SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.utils.context import reset_context
from experiments.fp32_residual_candidate import RMSNorm as Candidate
from experiments.fp16_residual_reference import RMSNorm as Baseline

torch.backends.cuda.matmul.allow_tf32=False
model=os.environ.get('NANO_MODEL', '/root/autodl-tmp/shared-context/models/Qwen3-0.6B-fp16')
llm=LLM(model,enforce_eager=True,max_model_len=2048,max_num_batched_tokens=2048,max_num_seqs=4,
 gpu_memory_utilization=.35,max_kvcache_blocks=64)
hf=AutoModelForCausalLM.from_pretrained(model,torch_dtype=torch.float32,attn_implementation='eager').eval().cuda()
prompts=[llm.tokenizer.encode('Explain how reference counting protects shared KV cache pages. '*n) for n in [2,12,32]]
for n in [128,513,1023]:
 rng=random.Random(987+n);prompts.append([rng.randrange(100,10000) for _ in range(n)])
original=RMSNorm.forward;original_hybrid=RMSNorm.hybrid_forward
rows=[]
with torch.inference_mode():
 for i,prompt in enumerate(prompts):
  ref=hf(torch.tensor([prompt],device='cuda')).logits[0,-1].float()
  for mode in ['fp16_residual','fp32_residual']:
   RMSNorm.forward=Baseline.forward if mode=='fp16_residual' else Candidate.forward
   RMSNorm.hybrid_forward=Baseline.hybrid_forward if mode=='fp16_residual' else Candidate.hybrid_forward
   llm.scheduler=Scheduler(llm.config);sc=llm.scheduler
   seq=Sequence(prompt,SamplingParams(max_tokens=1,ignore_eos=True));sc.add(seq)
   seqs,prefill=sc.schedule();ids,pos=llm.model_runner.prepare_prefill(seqs)
   got=llm.model_runner.run_model(ids,pos,True)[0].float()
   rel=((got-ref).norm()/ref.norm()).item()
   kl=(ref.softmax(-1)*(ref.log_softmax(-1)-got.log_softmax(-1))).sum().clamp_min(0).item()
   rows.append(dict(prompt=i,tokens=len(prompt),mode=mode,relative_l2=rel,kl=kl,argmax_match=got.argmax().item()==ref.argmax().item()))
   sc.postprocess(seqs,[100],True);reset_context();sc.block_manager.assert_invariants()
RMSNorm.forward=original;RMSNorm.hybrid_forward=original_hybrid
Path('results/finalization/hf-fp32-reference.json').write_text(json.dumps(dict(reference='Hugging Face eager attention, FP32 weights/activations, TF32 disabled',cases=rows),indent=2))
print(json.dumps(rows,indent=2));llm.exit()
