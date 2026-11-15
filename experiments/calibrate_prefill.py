from nanovllm.engine.schedule_cost import runtime_signature
"""Measure full runner Prefill (metadata, model, sampling) across query/context/batch shapes."""
import argparse,json,time,random,statistics
from pathlib import Path
import numpy as np
import torch
from nanovllm import LLM,SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.block_manager import BlockManager


def features(qs,cs):
    return [1.,float(any(cs)),sum(qs)/1024.,sum(q*c+q*(q+1)/2 for q,c in zip(qs,cs))/1048576.,len(qs)-1.]


def nnls(a,b):
    scale=np.maximum(np.linalg.norm(a,axis=0),1e-12);x=a/scale;w=np.zeros(a.shape[1]);res=b.copy()
    for _ in range(3000):
        old=w.copy()
        for j in range(len(w)):
            res+=x[:,j]*w[j];w[j]=max(0.,float(x[:,j]@res)/(float(x[:,j]@x[:,j])+1e-12));res-=x[:,j]*w[j]
        if np.max(np.abs(w-old))<1e-8:break
    return (w/scale).tolist()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--decode-profile',required=True)
    ap.add_argument('--output',required=True);args=ap.parse_args()
    llm=LLM(args.model,max_num_seqs=64,max_model_len=8192,max_num_batched_tokens=16384,
            gpu_memory_utilization=.84,shared_attention_backend='auto')
    runner=llm.model_runner;rng=random.Random(193);rows=[]
    shapes=[([q]*b,[c]*b,'train') for b in (1,2,4,8) for q in (256,512,1024,2048,4096)
            for c in (0,1024,3072) if q*b<=8192]
    shapes += [([q]*b,[c]*b,'holdout') for b,q,c in ((1,768,0),(1,1536,2048),(2,1536,512),(3,1024,2048),(4,768,0),(2,3072,1024))]
    shapes += [([256,1024,2048],[0,1024,3072],'holdout'),([512,512,1024,2048],[2048,0,1024,0],'holdout')]
    random.Random(193).shuffle(shapes)
    for qs,cs,split in shapes:
        bm=BlockManager(len(llm.scheduler.block_manager.blocks),256);bm.copy_blocks=runner.copy_kv_blocks
        seqs=[]
        for q,c in zip(qs,cs):
            seq=Sequence([rng.randrange(100,10000) for _ in range(q+c)],SamplingParams(max_tokens=1,ignore_eos=True))
            bm.allocate(seq,0)
            if c:
                seq.num_scheduled_tokens=c;runner.call('run',[seq],True)
            seq.num_cached_tokens=c;seq.num_scheduled_tokens=q;seqs.append(seq)
        samples=[]
        for repeat in range(7):
            torch.cuda.synchronize();t=time.perf_counter();runner.call('run',seqs,True)
            duration=(time.perf_counter()-t)*1000
            if repeat>=2:samples.append(duration)
        row=dict(q=qs,cached=cs,split=split,ms=statistics.median(samples),samples_ms=samples,features=features(qs,cs));rows.append(row)
        for seq in seqs:bm.deallocate(seq)
        bm.assert_invariants()
        print(qs,cs,split,row['ms'],flush=True)
    train=[r for r in rows if r['split']=='train'];a=np.array([r['features'] for r in train]);b=np.array([r['ms'] for r in train]);coeff=nnls(a,b)
    for r in rows:
        r['predicted_ms']=float(np.dot(r['features'],coeff));r['relative_error']=abs(r['predicted_ms']/r['ms']-1)
    profile=json.loads(Path(args.decode_profile).read_text())
    assert profile.get("runtime_signature")==runtime_signature(llm.config), "decode profile must be recalibrated"
    profile['prefill']=dict(coefficients=coeff,features=['intercept','cached_path','query_k','attention_m','batch_minus_one'],
        max_total_q=8192,max_cached=3072,max_batch=8,rows=rows,version=1,
        train_mape=statistics.mean(r['relative_error'] for r in train),
        holdout_mape=statistics.mean(r['relative_error'] for r in rows if r['split']=='holdout'))
    profile['runtime_signature']=runtime_signature(llm.config)
    Path(args.output).write_text(json.dumps(profile,indent=2));print(json.dumps({k:v for k,v in profile['prefill'].items() if k!='rows'}),flush=True);llm.exit()

if __name__=='__main__':main()
