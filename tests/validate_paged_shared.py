import argparse
import json
import math
from pathlib import Path
import torch
from flash_attn import flash_attn_with_kvcache
from nanovllm.layers.shared_prefix import shared_prefix_decode
from nanovllm.engine.shared_plan import select_shared_plan


def make_case(b, p, lengths, dtype, h=16, hk=8, d=128, padded=0, page=256):
    counts=[math.ceil(s/page) for s in lengths]
    n=p//page+sum(counts)+1
    k=torch.randn(n,page,hk,d,device='cuda',dtype=dtype)
    v=torch.randn_like(k)
    ids=torch.randperm(n,device='cuda',dtype=torch.int64).cpu().tolist()
    shared=ids[:p//page]
    cursor=p//page
    width=max(1,p//page+max(counts))
    bt=torch.full((b,width),-1,device='cuda',dtype=torch.int32)
    for i,count in enumerate(counts):
        row=shared+ids[cursor:cursor+count];cursor+=count
        if row: bt[i,:len(row)]=torch.tensor(row,device='cuda',dtype=torch.int32)
    lens=torch.tensor([p+s for s in lengths],device='cuda',dtype=torch.int32)
    if padded: lens[-padded:]=0
    # Non-contiguous query rows and head strides, as can occur after QKV splitting.
    storage=torch.randn(b,h*2,d,device='cuda',dtype=dtype)
    q=storage[:,::2,:]
    pref=torch.tensor([p],device='cuda',dtype=torch.int32)
    st=torch.full_like(bt,-1)
    st[:,:width-p//page]=bt[:,p//page:]
    sl=(lens-p).clamp_min(0)
    return q,k,v,bt,lens,pref,st,sl,d**-0.5


def verify(case, output=None):
    q,k,v,bt,lens,pref,st,sl,scale=case
    got=shared_prefix_decode(*case) if output is None else output
    expected=flash_attn_with_kvcache(q.unsqueeze(1),k,v,cache_seqlens=lens,
                                    block_table=bt,softmax_scale=scale,causal=True)
    tol=0.003 if q.dtype==torch.float16 else 0.02
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got,expected,atol=tol,rtol=tol)
    reference_error=0.0
    active=[i for i,l in enumerate(lens.tolist()) if l>0]
    for i in sorted(set(active[:1]+active[-1:])):
        length=int(lens[i]); pages=bt[i,:math.ceil(length/k.shape[1])].long()
        kd=k[pages].flatten(0,1)[:length].repeat_interleave(q.shape[1]//k.shape[2],1)
        vd=v[pages].flatten(0,1)[:length].repeat_interleave(q.shape[1]//k.shape[2],1)
        score=torch.einsum('hd,nhd->hn',q[i].float(),kd.float())*scale
        ref=torch.einsum('hn,nhd->hd',score.softmax(-1),vd.float())
        torch.testing.assert_close(got[i,0].float(),ref,atol=tol,rtol=tol)
        reference_error=max(reference_error,(got[i,0].float()-ref).abs().max().item())
    return {'fa2_max_abs':(got.float()-expected.float()).abs().max().item(),
            'fp32_max_abs':reference_error}


def planner_tests():
    class Seq:
        def __init__(self,pages,n): self.block_table,self.n=pages,n
        def __len__(self): return self.n
    seqs=[Seq([9,2,7,4,10+i],1025+i) for i in range(32)]
    assert select_shared_plan(seqs,256,'auto').prefix_blocks==4
    assert select_shared_plan(seqs[:31],256,'auto').reason=='small_batch'
    assert select_shared_plan(seqs,256,'off').reason=='off'
    assert select_shared_plan(seqs,256,'auto',graph_mode=False).reason=='eager_fallback'
    assert select_shared_plan(seqs,256,'force',supported=False).reason=='unsupported'
    assert select_shared_plan(seqs[:1],256,'force').prefix_blocks==4
    seqs[0].block_table[0]=99
    assert select_shared_plan(seqs,256,'auto').reason=='no_common_prefix'
    assert select_shared_plan([Seq([1,2],257)]*32,256,'auto').reason=='short_prefix'
    assert select_shared_plan([Seq([1],1)]*32,256,'force').reason=='no_common_prefix'
    long=[Seq([9,2,7,4]+list(range(20+i*8,28+i*8)),3000) for i in range(32)]
    assert select_shared_plan(long,256,'auto').reason=='suffix_dominates'
    return 'passed'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    torch.manual_seed(20261113)
    report={'planner':planner_tests(),'cases':[],'graph_replay':[]}
    for dtype in (torch.float16,torch.bfloat16):
        for b in (1,7,32,33):
            for p in (0,256,1024,4096):
                lengths=[(0,1,255,256,257,511,1025)[i%7] for i in range(b)]
                if b==1: lengths=[257]
                case=make_case(b,p,lengths,dtype,padded=2 if b==33 else 0)
                report['cases'].append(dict(dtype=str(dtype),batch=b,prefix=p,**verify(case)))
    for h,hk,d in ((8,2,64),(8,8,64)):
        case=make_case(7,256,[0,1,17,255,256,257,513],torch.bfloat16,h,hk,d)
        report['cases'].append(dict(heads=h,kv_heads=hk,dim=d,**verify(case)))
    case=make_case(7,1024,[0,1,511,512,513,1025,77],torch.float16,page=512)
    report['cases'].append(dict(page_size=512,**verify(case)))
    # All-empty sequence and zero suffix cases are independently exercised.
    for p in (0,256):
        case=make_case(4,p,[0]*4,torch.float16)
        report['cases'].append(dict(all_empty_suffix=True,prefix=p,**verify(case)))
    for dtype in (torch.float16,torch.bfloat16):
        case=make_case(33,2048,[1+i*17 for i in range(33)],dtype,padded=1)
        q,k,v,bt,lens,pref,st,sl,scale=case
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            shared_prefix_decode(*case)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream): out=shared_prefix_decode(*case)
        for p in (2048,1024,0,2048):
            pref.fill_(p);st.fill_(-1);st[:,:bt.shape[1]-p//256]=bt[:,p//256:]
            sl.copy_((lens-p).clamp_min(0));q.copy_(torch.randn_like(q));k.add_(0.001)
            graph.replay();torch.cuda.synchronize()
            report['graph_replay'].append(dict(dtype=str(dtype),prefix=p,**verify(case,out)))
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps({'cases':len(report['cases']),'graph_replays':len(report['graph_replay']),
                      'max_error':max(x['fa2_max_abs'] for x in report['cases']), 'planner':report['planner']}))


if __name__=='__main__': main()
