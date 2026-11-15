"""Preserve PyTorch reduction/rsqrt; fuse only bitwise-equivalent elementwise work."""
import torch
from torch import nn
import triton
import triton.language as tl

@triton.jit
def _prepare(X,R,V,S,N:tl.constexpr,H:tl.constexpr,ROWS:tl.constexpr,XS0:tl.constexpr,XS1:tl.constexpr,RS0:tl.constexpr,ADD:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);row=i//H;col=i%H
 x=tl.load(X+(row//ROWS)*XS0+(row%ROWS)*XS1+col,i<N,0).to(tl.float32)
 if ADD:x+=tl.load(R+row*RS0+col,i<N,0).to(tl.float32)
 tl.store(V+i,x,i<N);tl.store(S+i,x*x,i<N)

@triton.jit
def _finish(V,I,W,Y,R,N:tl.constexpr,H:tl.constexpr,ADD:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);row=i//H;col=i%H
 value=tl.load(V+i,i<N,0);inv=tl.load(I+row,i<N,0)
 rounded=(value*inv).to(Y.dtype.element_ty).to(tl.float32)
 weight=tl.load(W+col,i<N,0).to(tl.float32)
 tl.store(Y+i,rounded*weight,i<N)
 if ADD:tl.store(R+i,value,i<N)

class RMSNorm(nn.Module):
 def __init__(self,hidden_size,eps=1e-6):
  super().__init__();self.eps=eps;self.weight=nn.Parameter(torch.ones(hidden_size));self.graph_optimized=False
 def forward(self,x,residual=None):
  if self.graph_optimized:
   return self.hybrid_forward(x,residual)
  v=x.float() if residual is None else x.float()+residual.float()
  y=(v*torch.rsqrt(v.square().mean(-1,keepdim=True)+self.eps)).to(x.dtype)*self.weight
  return y if residual is None else (y,v.to(x.dtype))

 def hybrid_forward(self,x,residual=None):
  if x.device.type!='cuda' or x.ndim not in (2,3) or x.stride(-1)!=1 or (residual is not None and x.ndim!=2):
   v=x.float() if residual is None else x.float()+residual.float()
   y=(v*torch.rsqrt(v.square().mean(-1,keepdim=True)+self.eps)).to(x.dtype)*self.weight
   return y if residual is None else (y,v.to(x.dtype))
  n=x.numel();h=x.shape[-1]
  value=torch.empty(x.shape,dtype=torch.float32,device=x.device);square=torch.empty_like(value)
  rows=x.shape[1] if x.ndim==3 else 1
  _prepare[(triton.cdiv(n,256),)](x,residual if residual is not None else x,value,square,n,h,rows,x.stride(0),x.stride(1) if x.ndim==3 else 0,
    residual.stride(0) if residual is not None else 0,residual is not None,256,enable_fp_fusion=False)
  inv=torch.rsqrt(square.mean(-1,keepdim=True)+self.eps)
  y=torch.empty(x.shape,device=x.device,dtype=x.dtype);r=torch.empty_like(y) if residual is not None else y
  _finish[(triton.cdiv(n,256),)](value,inv,self.weight,y,r,n,h,residual is not None,256,enable_fp_fusion=False)
  return y if residual is None else (y,r)
