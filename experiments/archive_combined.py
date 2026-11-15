import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]; out=root/'results/combined'
old=json.loads((root/'results/cow/manifest.json').read_text())['sha256']
engine={name:hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in old.items() if name.startswith('nanovllm/')}
assert all(engine.values()),engine
files=[]
for folder in ('nanovllm','tests','experiments','docs','results/combined'):
    files += [p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json']
manifest=dict(engine_unchanged_since_cow=True,engine_files_checked=len(engine),
              branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),
              upstream=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
              sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
summary=json.loads((out/'summary.json').read_text())
print(json.dumps(dict(engine_unchanged=True,files=len(files),quality=summary['quality']),indent=2))
