"""Archive code/data, verify both reconstruction patches, and hash every included file."""
import ast,hashlib,json,shutil,subprocess,tarfile,tempfile
from pathlib import Path
base=Path('/root/autodl-tmp/shared-context');repo=base/'src/nano-vllm-shared';out=repo/'results/finalization';prov=out/'environment';prov.mkdir(exist_ok=True)
readme=repo/'README.md';link='\nShared-context extension: [implementation, final results and reproduction](README.shared-context.md).\n'
if 'README.shared-context.md' not in readme.read_text():readme.write_text(readme.read_text()+link)
for name in ['release-progress.log','supplement-progress.log','final-checks-progress.log']:
 shutil.copy2(Path('/root')/name,out/name)
shutil.copy2(base/'run_cow_env.sh',prov/'run_cow_env.sh')
freeze=subprocess.check_output([str(base/'envs/main/bin/python'),'-m','pip','freeze','--path',str(base/'envs/flashinfer-reference')],text=True)
(prov/'flashinfer-requirements.txt').write_text(freeze)
source_files=lambda root:[p for p in root.rglob('*') if p.is_file() and not any(x in p.relative_to(root).parts for x in ['.git','__pycache__','results','assets']) and (p.suffix in ('.py','.sh','.md','.toml') or p.name in ('LICENSE','.gitignore'))]
patches=out/'patches';patches.mkdir(exist_ok=True)
for label,before_root in [('from-upstream',base/'src/nano-vllm'),('from-pre-finalization',base/'src/nano-vllm-pre-finalization')]:
 with tempfile.TemporaryDirectory(prefix='release-patch-') as tmp:
  tmp=Path(tmp)
  for name,src in [('before',before_root),('after',repo)]:
   for p in source_files(src):
    dst=tmp/name/p.relative_to(src);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
  result=subprocess.run(['git','diff','--no-index','--no-renames','--binary','--src-prefix=a/','--dst-prefix=b/','before','after'],cwd=tmp,stdout=subprocess.PIPE)
  assert result.returncode in (0,1)
  patchlines=[]
  for line in result.stdout.splitlines(keepends=True):
   if line.startswith((b'diff --git ',b'--- ',b'+++ ')):
    for old,new in [(b'a/before/',b'a/'),(b'b/after/',b'b/'),(b'a/after/',b'a/'),(b'b/before/',b'b/')]:
     line=line.replace(old,new)
   patchlines.append(line)
  patch=b''.join(patchlines)
  path=patches/(label+'.patch');path.write_bytes(patch)
  subprocess.run(['git','apply','--check',str(path)],cwd=tmp/'before',check=True)
  subprocess.run(['git','apply',str(path)],cwd=tmp/'before',check=True)
  expected={str(p.relative_to(tmp/'after')):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files(tmp/'after')}
  actual={str(p.relative_to(tmp/'before')):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files(tmp/'before')}
  if expected!=actual:
   print("PATCH MISMATCH",label,"missing",sorted(set(expected)-set(actual)),"extra",sorted(set(actual)-set(expected)),"changed",[k for k in expected.keys() & actual.keys() if expected[k]!=actual[k]],flush=True)
  assert expected==actual,label
(out/'patch-verification.json').write_text(json.dumps(dict(upstream=True,pre_finalization=True,reconstructed_source_matches=True),indent=2))
# Validate syntax without importing dependencies or generating bytecode.
pycount=0
for p in source_files(repo):
 if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p));pycount+=1
subprocess.run(['git','diff','--check'],cwd=repo,check=True)
subprocess.run(['bash','-n',*[str(p) for p in (repo/'experiments').glob('*.sh')]],check=True)
files=[p for p in repo.rglob('*') if p.is_file() and not any(x in p.relative_to(repo).parts for x in ['.git','__pycache__']) and p.name!='SHA256SUMS.json']
manifest={str(p.relative_to(repo)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)}
manifestpath=repo/'SHA256SUMS.json';manifestpath.write_text(json.dumps(manifest,indent=2))
archive=base/'shared-context-final-2026-11-15.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for p in sorted(files)+[manifestpath]:tar.add(p,arcname=p.relative_to(repo),recursive=False)
sha=hashlib.sha256(archive.read_bytes()).hexdigest();archive.with_suffix(archive.suffix+'.sha256').write_text(sha+'  '+archive.name+'\n')
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha,files=len(files),python_files=pycount,patches_verified=2),indent=2))
