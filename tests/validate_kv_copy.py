import json
import torch
from nanovllm.engine.kv_copy import copy_kv_pages

rows=[]
for dtype in (torch.float16,torch.bfloat16):
    torch.manual_seed(10)
    cache=torch.full((2,3,6,256,8,128),-8.,device='cuda',dtype=dtype)
    cache[:,:,0]=torch.randn_like(cache[:,:,0]); cache[:,:,1]=torch.randn_like(cache[:,:,1])
    sources=cache[:,:,:2].clone()
    copy_kv_pages(cache,[(0,2,1),(0,3,255),(1,4,256),(1,5,0)])
    for src,dst,count in [(0,2,1),(0,3,255),(1,4,256),(1,5,0)]:
        assert torch.equal(cache[:,:,dst,:count],cache[:,:,src,:count])
        assert torch.all(cache[:,:,dst,count:]==-8)
    assert torch.equal(sources,cache[:,:,:2])
    rows.append(dict(dtype=str(dtype),bitwise=True,padding_untouched=True,source_unchanged=True))
print(json.dumps(dict(passed=True,cases=rows)))
