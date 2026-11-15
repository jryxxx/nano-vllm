import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _rms_kernel(X, R, W, Y, O, H: tl.constexpr, ROWS: tl.constexpr,
                XS0: tl.constexpr, XS1: tl.constexpr, RS0: tl.constexpr,
                EPS: tl.constexpr, ADD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    offset = (row // ROWS)*XS0 + (row % ROWS)*XS1 + col
    x = tl.load(X+offset, col<H, 0).to(tl.float32)
    if ADD:
        x += tl.load(R+row*RS0+col, col<H, 0).to(tl.float32)
        tl.store(O+row*H+col, x, col<H)
    variance = tl.sum(x*x, 0)/H
    normalized = x*tl.rsqrt(variance+EPS)
    # Match eager's explicit rounding BEFORE multiplying the learned weight.
    # torch.compile may fuse away this materialization and alter model logits.
    normalized = normalized.to(Y.dtype.element_ty).to(tl.float32)
    weight = tl.load(W+col, col<H, 0).to(tl.float32)
    tl.store(Y+row*H+col, normalized*weight, col<H)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, residual=None):
        if x.device.type != 'cuda' or x.ndim not in (2,3) or x.stride(-1)!=1 or (residual is not None and x.ndim!=2):
            value=x.float() if residual is None else x.float()+residual.float()
            norm=(value*torch.rsqrt(value.square().mean(-1,keepdim=True)+self.eps)).to(x.dtype)*self.weight
            return norm if residual is None else (norm,value.to(x.dtype))
        width=x.shape[-1]
        y=torch.empty(x.shape,device=x.device,dtype=x.dtype)
        out=torch.empty_like(y) if residual is not None else y
        rows=x.shape[1] if x.ndim==3 else 1
        _rms_kernel[(x.numel()//width,)](x,residual if residual is not None else x,self.weight,y,out,
            width,rows,x.stride(0),x.stride(1) if x.ndim==3 else 0,
            residual.stride(0) if residual is not None else 0,self.eps,residual is not None,
            triton.next_power_of_2(width),num_warps=4)
        return y if residual is None else (y,out)
