"""Fixed-shape complete Decode paths: FlashAttention, custom split/merge, FlashInfer Cascade."""
import json,time,statistics,math,argparse
from pathlib import Path
import torch,flashinfer
from flash_attn import flash_attn_with_kvcache
from flashinfer.cascade import BatchDecodeWithSharedPrefixPagedKVCacheWrapper
from nanovllm.layers.shared_prefix import shared_prefix_decode

def measure(fn,graph):
 for _ in range(5):fn()
 torch.cuda.synchronize()
 if graph:
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):out=fn()
  call=g.replay
 else:call=fn
 samples=[]
 for _ in range(3):
  torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  wall=time.perf_counter();start.record()
  for j in range(100):call()
  end.record();end.synchronize()
  samples.append(dict(gpu_us=start.elapsed_time(end)*10,wall_us=(time.perf_counter()-wall)*10000))
 return samples

ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--output',default='results/finalization/flashinfer-reference.json');args=ap.parse_args()
torch.manual_seed(20261115);rows=[]
shapes=[(b,p,s,16) for b in (8,32,64) for p in (1024,4096) for s in (128,1024)]
shapes += [(b,p,128,32) for b in (32,64) for p in (1024,4096)]
if args.smoke:shapes=shapes[:1]
for b,p,s,h in shapes:
 page=256;hk=8;d=128;pp=p//page;sp=(s+page-1)//page
 q=torch.randn(b,h,d,device='cuda',dtype=torch.float16)
 k=torch.randn(pp+b*sp,page,hk,d,device='cuda',dtype=torch.float16);v=torch.randn_like(k)
 suffix=torch.arange(pp,pp+b*sp,device='cuda',dtype=torch.int32).view(b,sp)
 bt=torch.cat([torch.arange(pp,device='cuda',dtype=torch.int32).expand(b,-1),suffix],1).contiguous()
 lengths=torch.full((b,),p+s,device='cuda',dtype=torch.int32);sl=torch.full((b,),s,device='cuda',dtype=torch.int32)
 prefix=torch.tensor([p],device='cuda',dtype=torch.int32)
 workspace=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
 wrapper=BatchDecodeWithSharedPrefixPagedKVCacheWrapper(workspace,'NHD')
 start=time.perf_counter()
 wrapper.begin_forward(torch.arange(0,(b+1)*sp,sp,device='cuda',dtype=torch.int32),suffix.flatten(),
  torch.full((b,),(s-1)%page+1,device='cuda',dtype=torch.int32),h,hk,d,page,data_type='float16')
 torch.cuda.synchronize();planning_ms=(time.perf_counter()-start)*1000
 calls={
  'flash_attention':lambda:flash_attn_with_kvcache(q.unsqueeze(1),k,v,cache_seqlens=lengths,block_table=bt,softmax_scale=d**-.5,causal=True).squeeze(1),
  'custom_shared':lambda:shared_prefix_decode(q,k,v,bt,lengths,prefix,suffix,sl,d**-.5).squeeze(1),
  'flashinfer_cascade':lambda:wrapper.forward(q,k[:pp].view(p,hk,d),v[:pp].view(p,hk,d),(k,v))}
 references=[]
 for i in range(min(4,b)):
  kk=k[bt[i].long()].reshape(-1,hk,d)[:p+s].repeat_interleave(h//hk,1).float()
  vv=v[bt[i].long()].reshape(-1,hk,d)[:p+s].repeat_interleave(h//hk,1).float()
  scores=torch.einsum('hd,lhd->hl',q[i].float(),kk)*d**-.5
  references.append(torch.einsum('hl,lhd->hd',scores.softmax(-1),vv))
 reference=torch.stack(references)
 for name,fn in calls.items():
  got=fn()[:len(reference)].float();torch.testing.assert_close(got,reference,atol=.003,rtol=.003)
  rel=((got-reference).norm()/reference.norm()).item()
  for graph in (False,True):
   samples=measure(fn,graph);row=dict(batch=b,prefix=p,suffix=s,heads=h,backend=name,graph=graph,relative_l2=rel,
     gpu_us=statistics.median(x['gpu_us'] for x in samples),samples=samples,flashinfer_planning_ms=planning_ms)
   rows.append(row);print(b,p,s,h,name,graph,round(row['gpu_us'],2),flush=True)
   Path(args.output).write_text(json.dumps(dict(flashinfer_version=flashinfer.__version__,
     note='Complete attention paths including merge; fixed metadata planning excluded and reported separately; contiguous shared prefix; correctness checks first four queries against FP32.',rows=rows),indent=2))
