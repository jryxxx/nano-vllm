import difflib
import hashlib
import json
import subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[1]; previous=root.parent/'nano-vllm-pre-cow'; out=root/'results/cow'
paths=[]
for directory in ('nanovllm','tests','experiments','docs'):
    paths.extend(p for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.md','.sh'))
patch=[];changed=[]
for p in sorted(paths):
    rel=p.relative_to(root); before=previous/rel
    old=before.read_text() if before.exists() else ''
    new=p.read_text()
    if old!=new:
        changed.append(str(rel))
        patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
                     fromfile='a/'+str(rel) if before.exists() else '/dev/null',tofile='b/'+str(rel)))
(out/'phase-cow.patch').write_text(''.join(patch))
files=paths+list(out.glob('*.json'))+[out/'phase-cow.patch']
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files) if p.name!='manifest.json'}
(out/'manifest.json').write_text(json.dumps(dict(branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),
    upstream=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),changed_since_pre_cow=changed,sha256=manifest),indent=2))
print(json.dumps(dict(changed_files=len(changed),hashed_files=len(manifest))))
