"""Retest the rejected fusion under full-precision GEMM reduction; no production mutation."""
import argparse,json,subprocess,sys,runpy
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['cache','prefill']);args=ap.parse_args()
out=Path('results/finalization')
if args.case:
 from nanovllm.layers.layernorm import RMSNorm
 from experiments.fp32_residual_candidate import RMSNorm as Candidate
 RMSNorm.forward=Candidate.forward
 RMSNorm.hybrid_forward=Candidate.hybrid_forward
 if args.case=='cache':
  path=out/'fp32-residual-cache.json'
  sys.argv=['experiments/bench_cache.py','--mode','validate','--output',str(path)]
  runpy.run_path(sys.argv[0],run_name='__main__')
  if not json.loads(path.read_text())['full_recompute_gate_passed']:raise SystemExit(2)
 else:
  sys.argv=['tests/validate_prefill_gpu.py','--model','/root/autodl-tmp/shared-context/models/Qwen3-0.6B-fp16',
   '--profile',str(out/'cost-profile.json'),'--output',str(out/'fp32-residual-prefill.json')]
  runpy.run_path(sys.argv[0],run_name='__main__')
else:
 results=[]
 for case in ['cache','prefill']:
  with (out/f'fp32-residual-{case}.log').open('w') as log:
   p=subprocess.run([sys.executable,__file__,'--case',case],stdout=log,stderr=subprocess.STDOUT)
  results.append(dict(case=case,returncode=p.returncode));print(case,p.returncode,flush=True)
 (out/'fp32-residual-result.json').write_text(json.dumps(dict(passed=all(r['returncode']==0 for r in results),cases=results),indent=2))
