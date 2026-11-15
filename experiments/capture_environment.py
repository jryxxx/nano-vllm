import subprocess,json,hashlib,platform,sys
from pathlib import Path
root=Path('/root/autodl-tmp/shared-context');repo=root/'src/nano-vllm-shared';out=repo/'results/finalization/environment';out.mkdir(exist_ok=True)
def run(cmd):return subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT).strip()
report=dict(platform=platform.platform(),upstream_commit=run(['git','-C',str(root/'src/nano-vllm'),'rev-parse','HEAD']),gpu=run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader']),nvcc=run(['/usr/local/cuda/bin/nvcc','--version']),python={})
for name in ['main','vllm-reference']:
 py=str(root/'envs'/name/'bin/python');report['python'][name]=run([py,'--version']);(out/(name+'-requirements.txt')).write_text(run([py,'-m','pip','freeze'])+'\n')
manifest={}
for name in ['Qwen3-0.6B','Qwen3-0.6B-fp16','Qwen3-4B','Qwen3-4B-fp16']:
 d=root/'models'/name;manifest[name]={}
 for p in sorted(d.iterdir()):
  if p.is_file():
   digest=hashlib.sha256()
   with p.open('rb') as f:
    for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
   manifest[name][p.name]=dict(bytes=p.stat().st_size,sha256=digest.hexdigest(),symlink=p.is_symlink())
(out/'models.json').write_text(json.dumps(manifest,indent=2));(out/'environment.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
