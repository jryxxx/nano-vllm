import difflib,hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1];before=root.parent/'nano-vllm-pre-prefill-cost';out=root/'results/prefill'
paths=[]
for folder in ('nanovllm','tests','experiments','docs'):
    paths.extend(p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.md','.sh'))
patch=[];changed=[]
for p in sorted(paths):
    rel=p.relative_to(root);old=before/rel;a=old.read_text() if old.exists() else '';b=p.read_text()
    if a!=b:
        changed.append(str(rel));patch.extend(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/'+str(rel) if old.exists() else '/dev/null',tofile='b/'+str(rel)))
(out/'phase-prefill.patch').write_text(''.join(patch))
files=paths+[p for p in out.iterdir() if p.is_file() and p.name!='manifest.json']+[root/'results/scheduler/cost-profile.json']
manifest=dict(branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),
              upstream=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),changed_since_previous_phase=changed,
              sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(changed=len(changed),hashed=len(files))))
