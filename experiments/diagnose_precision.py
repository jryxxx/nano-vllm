import torch,runpy,sys
mode=sys.argv[1]
if mode=='matmul':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
elif mode=='residual':
 from nanovllm.layers.layernorm import RMSNorm
 def add_rms(self,x,residual):
  value=x.float()+residual.float()
  norm=(value*torch.rsqrt(value.square().mean(-1,keepdim=True)+self.eps)).to(x.dtype)*self.weight
  return norm,value
 RMSNorm.add_rms_forward=add_rms
sys.argv=['tests/validate_prefill_gpu.py','--model','/root/autodl-tmp/shared-context/models/Qwen3-0.6B-fp16','--profile','results/prefill/cost-profile.json','--output',f'results/finalization/precision-{mode}.json']
runpy.run_path('tests/validate_prefill_gpu.py',run_name='__main__')
