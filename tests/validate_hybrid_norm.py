import json,torch
from nanovllm.layers.layernorm import RMSNorm
checks=0
for dtype in [torch.float16,torch.bfloat16]:
 for n in [1,3,32,257,1024]:
  for h in [128,1024,2560]:
   for add in [False,True]:
    for scale in [1e-3,1.,10.]:
     for residual_dtype in ([dtype,torch.float32] if add else [dtype]):
      torch.manual_seed(190);x=torch.randn(n,h*2,device='cuda',dtype=dtype)[:,:h]*scale
      residual=torch.randn(x.shape,device='cuda',dtype=residual_dtype) if add else None
      mod=RMSNorm(h).to(device='cuda',dtype=dtype);mod.weight.data.uniform_(.5,1.5)
      v=x.float() if not add else x.float()+residual.float()
      expected=(v*torch.rsqrt(v.square().mean(-1,keepdim=True)+mod.eps)).to(dtype)*mod.weight
      for graph in [False,True]:
       mod.graph_optimized=graph
       with torch.no_grad():got=mod(x,residual)
       if add:
        assert got[1].dtype==torch.float32 and torch.equal(got[1],v);got=got[0]
       assert torch.equal(got,expected),(dtype,n,h,add,scale,graph,(got-expected).abs().max().item())
       checks+=1
 for n in [1,32,257]:
  x=torch.randn(n,32,128,device='cuda',dtype=dtype)[:,:16]
  mod=RMSNorm(128).to(device='cuda',dtype=dtype)
  v=x.float();expected=(v*torch.rsqrt(v.square().mean(-1,keepdim=True)+mod.eps)).to(dtype)*mod.weight
  for graph in [False,True]:
   mod.graph_optimized=graph
   with torch.no_grad():got=mod(x)
   assert torch.equal(got,expected);checks+=1
print(json.dumps(dict(passed=True,bitwise_cases=checks,residual_dtype='float32')))
