import concurrent.futures
import hashlib
import json
from pathlib import Path
import requests

root = Path('/root/autodl-tmp/shared-context')
dest = root / 'models/Qwen3-4B'
dest.mkdir(exist_ok=True, parents=True)
listing = json.loads((root / 'logs/model-4b-files.json').read_text())
files = [f for f in listing['Data']['Files'] if f['Type'] == 'blob']

def fetch(f):
    path = dest / f['Path']
    url = 'https://modelscope.cn/api/v1/models/Qwen/Qwen3-4B/repo'
    h = hashlib.sha256()
    with requests.get(url, params={'Revision': f['Revision'], 'FilePath': f['Path']},
                      stream=True, timeout=(20, 120)) as r:
        r.raise_for_status()
        with path.with_suffix(path.suffix + '.part').open('wb') as out:
            for chunk in r.iter_content(4 * 1024 * 1024):
                h.update(chunk)
                out.write(chunk)
    assert h.hexdigest() == f['Sha256'], (f['Path'], h.hexdigest())
    path.with_suffix(path.suffix + '.part').replace(path)
    print(f['Path'], f['Size'], h.hexdigest(), flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(fetch, files))
(dest / 'download-manifest.json').write_text(json.dumps(files, indent=2))
