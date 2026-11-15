"""Rebuild aggregate JSON/CSV and exportable figures from measured trials only."""
import json,csv,statistics,collections,argparse
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',default='results/finalization');args=ap.parse_args();root=Path(args.root)
def read(name):return json.loads((root/name).read_text())
def med(xs):return statistics.median(xs)
def aggregate(rows,keys,metrics):
 groups=collections.defaultdict(list)
 for row in rows:
  if row.get('repeat',0)>=0:groups[tuple(row[k] for k in keys)].append(row)
 out=[]
 for key,rs in groups.items():
  r=dict(zip(keys,key));r['n']=len(rs)
  for metric in metrics:
   vals=[x[metric] for x in rs if x.get(metric) is not None]
   if vals:r[metric]=med(vals);r[metric+'_min']=min(vals);r[metric+'_max']=max(vals)
  out.append(r)
 return out
allrows={}
rows=[]
for p in sorted(root.glob('sessions-*.json')):
 for r in json.loads(p.read_text())['trials']:
  ts=[v['token_times_s'] for v in r['requests_raw']];r['request_tpot_median_ms']=med((v[-1]-v[0])*1000/(len(v)-1) for v in ts)
  r['input_tokens_per_s']=r['input_tokens']/r['elapsed_s'];r['actual_prefill_tokens_per_s']=r['prefill_tokens']/r['elapsed_s'];rows.append(r)
allrows['sessions']=aggregate(rows,['scenario','variant'],['output_tokens_per_s','input_tokens_per_s','actual_prefill_tokens_per_s','prefill_tokens','ttft_p95_ms','request_max_itl_p95_ms','request_tpot_median_ms','goodput'])
if (root/'openloop-final.json').exists():
 rows=read('openloop-final.json')['trials']
 for r in rows:
  for cohort,c in r['cohorts'].items():r[cohort+'_ttft_p95_ms']=c['ttft_ms']['95'];r[cohort+'_max_itl_p95_ms']=c['request_max_itl_ms']['95']
  times=[v['token_times_s'] for v in r['records'] if len(v.get('token_times_s',[]))>1]
  r['request_tpot_median_ms']=med((v[-1]-v[0])*1000/(len(v)-1) for v in times) if times else None
 allrows['openloop']=aggregate(rows,['scenario','variant'],['output_tokens_per_s','goodput','accepted','rejected','unfinished','hot_ttft_p95_ms','cold_ttft_p95_ms','hot_max_itl_p95_ms','cold_max_itl_p95_ms','request_tpot_median_ms'])
rows=[]
for mode in ['off','auto','always']:
 for r in read(f'tier-{mode}.json')['trials']:
  for k in ['restored_pages','spilled_pages','h2d_bytes','d2h_bytes','pinned_bytes','staging_bytes']:r[k]=r['transfer'].get(k,0)
  rows.append(r)
allrows['tier']=aggregate(rows,['scenario','mode'],['output_tokens_per_s','ttft_p95_ms','unique_cached_logical_pages','gpu_kv_reserved_bytes','restored_pages','spilled_pages','h2d_bytes','d2h_bytes','pinned_bytes','staging_bytes'])
allrows['cache']=aggregate(read('cache-final.json')['trials'],['scenario','pages','policy'],['output_tokens_per_s','ttft_p95_ms'])
if (root/'combined-final.json').exists():allrows['combined']=aggregate(read('combined-final.json')['trials'],['scenario','variant'],['cold_tokens_per_s','branch_tokens_per_s','branch_first_ms','peak_live_pages','copy_bytes','planning_ms'])
rows=[]
for p in sorted(root.glob('reference-*.json')):
 d=json.loads(p.read_text())
 for r in d['trials']:r['model']='4B' if '4b' in p.name else '0.6B';rows.append(r)
if rows:allrows['external']=aggregate(rows,['model','scenario','backend'],['output_tokens_per_s','ttft_p95_ms','itl_p95_ms'])
allrows['attention']=read('flashinfer-reference.json')['rows']
(root/'summary.json').write_text(json.dumps(allrows,indent=2))
for name,rows in allrows.items():
 keys=list(dict.fromkeys(k for r in rows for k,v in r.items() if not isinstance(v,(dict,list))))
 with (root/(name+'-summary.csv')).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
# Plotting uses the isolated host matplotlib, not the inference environment.
try:import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
except ImportError:print('JSON/CSV written; install matplotlib to export figures.');raise SystemExit(0)
figdir=root/'figures';figdir.mkdir(exist_ok=True)
def save(fig,name):
 fig.tight_layout();fig.savefig(figdir/(name+'.png'),dpi=180);fig.savefig(figdir/(name+'.svg'));plt.close(fig)
def grouped(name,category,label,metric,ylabel,filename):
 rows=allrows[name];cats=list(dict.fromkeys(r[category] for r in rows));labs=list(dict.fromkeys(r[label] for r in rows))
 fig,axes=plt.subplots(1,len(cats),figsize=(max(8,4*len(cats)),4),squeeze=False)
 for ax,cat in zip(axes[0],cats):
  rr=[r for r in rows if r[category]==cat];xs=range(len(rr));vs=[r[metric] for r in rr]
  lo=[v-r.get(metric+'_min',v) for r,v in zip(rr,vs)];hi=[r.get(metric+'_max',v)-v for r,v in zip(rr,vs)]
  ax.bar(xs,vs,yerr=[lo,hi],capsize=2,color='#3778ad');ax.set_xticks(list(xs),[r[label] for r in rr],rotation=60,ha='right');ax.set_title(str(cat));ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2)
 save(fig,filename)
grouped('sessions','scenario','variant','output_tokens_per_s','Output tokens / second (median, min-max)','session-ablation')
grouped('openloop','scenario','variant','goodput','SLO-qualified requests / second','openloop-goodput')
grouped('tier','scenario','mode','output_tokens_per_s','Output tokens / second','cpu-tier')
if 'combined' in allrows:grouped('combined','scenario','variant','cold_tokens_per_s','Output tokens / second (cold parent included)','branch-ablation')
if 'external' in allrows:
 for r in allrows['external']:r['case']=r['model']+' '+r['scenario']
 grouped('external','case','backend','output_tokens_per_s','Output tokens / second','external-reference')
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,heads in zip(axes,[16,32]):
 rs=[r for r in allrows['attention'] if r['graph'] and r['heads']==heads];shapes=list(dict.fromkeys((r['batch'],r['prefix'],r['suffix']) for r in rs))
 for backend in ['flash_attention','custom_shared','flashinfer_cascade']:
  vals=[next((r['gpu_us'] for r in rs if (r['batch'],r['prefix'],r['suffix'])==s and r['backend']==backend),float('nan')) for s in shapes]
  ax.plot(range(len(shapes)),vals,'o-',label=backend)
 ax.set_xticks(range(len(shapes)),[f'{b}/{p}/{s}' for b,p,s in shapes],rotation=65,ha='right');ax.set_title(f'Q heads={heads}; CUDA Graph');ax.set_ylabel('Complete attention GPU time (us)');ax.set_xlabel('Batch / prefix / suffix');ax.legend(fontsize=8);ax.grid(alpha=.2)
save(fig,'attention-reference')
print('Wrote',len(allrows),'tables and figures to',root)
