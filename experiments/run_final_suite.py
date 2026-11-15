"""Serial GPU suite with explicit logs and progress records; no overlapping benchmarks."""
import subprocess,json,os,hashlib,time,argparse,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'results/finalization';out.mkdir(exist_ok=True)
ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['sessions','cache','tier','reference','core'],required=True);args=ap.parse_args()
python=sys.executable;model=os.environ.get('NANO_MODEL','/root/autodl-tmp/shared-context/models/Qwen3-0.6B-fp16')
env=dict(os.environ,PYTHONPATH=str(root),PYTHONUNBUFFERED='1');jobs=[]
def job(name,script,options=(),extra=None):jobs.append((name,[python,script,*options],extra or {}))
if args.group=='sessions':
 for variant in ['B0','B1','B2','M','K','S','Full','Tiered']:
  job('sessions-'+variant,'experiments/replay_sessions.py',['--variant',variant,'--model',model,'--output',str(out/f'sessions-{variant}.json')],{'PYTHONPATH':os.environ.get('NANO_UPSTREAM',str(root/'vendor/nano-vllm-upstream'))} if variant=='B0' else {})
elif args.group=='cache':job('cache-final','experiments/bench_cache.py',['--mode','bench','--pages','20','32','64','--output',str(out/'cache-final.json')])
elif args.group=='tier':
 for mode in ['off','auto','always']:job('tier-'+mode,'experiments/bench_tier.py',['--mode',mode,'--output',str(out/f'tier-{mode}.json')])
 job('transfer-profile','experiments/profile_transfers.py')
elif args.group=='reference':
 job('flashinfer','experiments/bench_flashinfer.py',extra={'PYTHONPATH':os.environ.get('NANO_FLASHINFER_PATH','/root/autodl-tmp/shared-context/envs/flashinfer-reference')+':'+str(root),'MAX_JOBS':'4','TORCH_CUDA_ARCH_LIST':'8.9'})
elif args.group=='core':
 for name in ['validate_cow_metadata','validate_cache','validate_cache_policy','validate_scheduler']:
  job(name,'tests/'+name+'.py')
 job('prefill-model','tests/validate_prefill_model.py',['--model',model,'--profile',str(out/'cost-profile.json'),'--output',str(out/'prefill-model.json')])
 job('cache-logits','experiments/bench_cache.py',['--mode','validate','--output',str(out/'cache-logits.json')])
 job('prefill-gpu','tests/validate_prefill_gpu.py',['--model',model,'--profile',str(out/'cost-profile.json'),'--output',str(out/'prefill-gpu.json')])
 job('tier-validation','tests/validate_tier.py')
 job('cow-gpu','tests/validate_cow_gpu.py',['--model',model,'--backend','auto','--output',str(out/'cow-gpu.json')])
 job('resident-kv','tests/validate_cache_resident.py')
state=[]
for name,cmd,extra in jobs:
 digest=hashlib.sha256()
 for p in sorted((root/'nanovllm').rglob('*.py')):digest.update(p.read_bytes())
 row=dict(name=name,command=cmd,code_sha256=digest.hexdigest(),status='running',log=f'{name}.log');state.append(row)
 path=out/f'suite-{args.group}.json';path.write_text(json.dumps(state,indent=2));start=time.perf_counter()
 with (out/row['log']).open('w') as log:
  result=subprocess.run(cmd,cwd=root,env=dict(env,**extra),stdout=log,stderr=subprocess.STDOUT)
 row.update(status='passed' if result.returncode==0 else 'failed',exit_code=result.returncode,elapsed_s=time.perf_counter()-start)
 path.write_text(json.dumps(state,indent=2));print(name,row['status'],round(row['elapsed_s'],2),flush=True)
 if result.returncode:raise SystemExit(result.returncode)
