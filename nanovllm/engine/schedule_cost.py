"""Small measured lookup table; no uncalibrated extrapolation across routes."""
import json
import math
import hashlib
import torch
from pathlib import Path
from nanovllm.engine.shared_plan import select_shared_plan


def runtime_signature(config):
    return dict(precision='fp32-residual-graph-fusion-fullaccum-v4',
                model_config_sha256=hashlib.sha256((Path(config.model)/'config.json').read_bytes()).hexdigest(),
                torch=torch.__version__,cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu')


class ScheduleCost:
    def __init__(self, config):
        self.rows = []
        self.prefill_unit = None
        self.prefill = None
        self.prefill_correction = 1.0
        self.prefill_exact = {}
        if config.schedule_cost_profile:
            profile = json.loads(Path(config.schedule_cost_profile).read_text())
            if profile['model'] != config.model or profile['eager'] != config.enforce_eager:
                raise ValueError('scheduler cost profile model/execution mode mismatch')
            if profile.get('runtime_signature') != runtime_signature(config):
                raise ValueError('cost profile runtime/precision mismatch; recalibrate for this engine')
            self.rows = profile['decode']
            self.prefill = profile.get('prefill')
            if self.prefill:
                self.prefill_train = [r for r in self.prefill['rows'] if r['split']=='train']
                self.prefill_exact = {tuple(sorted(zip(r['q'],r['cached']))):r['ms'] for r in self.prefill_train}

    @staticmethod
    def features(seqs, config, capable):
        common = select_shared_plan(seqs, config.kvcache_block_size, 'force').prefix_blocks
        selected = select_shared_plan(seqs, config.kvcache_block_size, config.shared_attention_backend,
                                      config.shared_min_batch, config.shared_min_prefix, capable,
                                      not config.enforce_eager).enabled
        return dict(batch=len(seqs), context=sum(len(s) for s in seqs)/len(seqs),
                    common=bool(common), shared=selected, prefix_tokens=common*config.kvcache_block_size)

    def estimate(self, features):
        rows = [r for r in self.rows if r['common']==features['common'] and r['shared']==features['shared']]
        if not rows:
            return None
        def distance(r):
            return abs(math.log2(r['batch']/features['batch'])) + abs(math.log2(r['context']/features['context']))
        row = min(rows,key=distance)
        # Stay within calibrated shape neighborhood; otherwise use fair FCFS.
        if distance(row) > .6:
            return None
        return row['ms']

    def observe(self, features, ms):
        match = next((r for r in self.rows if r['batch']==features['batch']
                      and r['common']==features['common'] and r['shared']==features['shared']
                      and abs(math.log2(r['context']/features['context'])) < .15), None)
        if match is not None:
            match['ms'] = .9*match['ms']+.1*ms
        elif len(self.rows)<128:
            self.rows.append(dict(features,ms=ms))

    def prefill_budget(self, config):
        # Fixed launch/model costs make ms/token inversion unsafe. Until an
        # affine prefill model is calibrated, retain a bounded token chunk.
        return config.schedule_prefill_chunk

    def observe_prefill(self, tokens, context, ms):
        # Conservative recent cost per query token; includes context attention and
        # metadata. No assertion of a hard time bound for a non-preemptible kernel.
        unit=ms/max(tokens,1)
        self.prefill_unit=unit if self.prefill_unit is None else .8*self.prefill_unit+.2*unit

    @staticmethod
    def prefill_features(qs, cs):
        return [1.,float(any(cs)),sum(qs)/1024.,
                sum(q*c+q*(q+1)/2 for q,c in zip(qs,cs))/1048576.,len(qs)-1.]

    def predict_prefill(self, qs, cs):
        if not self.prefill or not qs or any(q<=0 for q in qs): return None
        if (sum(qs)>self.prefill['max_total_q'] or len(qs)>self.prefill['max_batch']
                or max(cs)>self.prefill['max_cached']): return None
        key=tuple(sorted(zip(qs,cs)))
        exact=self.prefill_exact.get(key)
        if exact is not None: return exact*self.prefill_correction
        features=self.prefill_features(qs,cs)
        base=sum(a*b for a,b in zip(features,self.prefill['coefficients']))
        mean_c=sum(q*c for q,c in zip(qs,cs))/sum(qs)
        def distance(row):
            rq,rc=row['q'],row['cached'];mean=sum(q*c for q,c in zip(rq,rc))/sum(rq)
            return (abs(math.log2(sum(rq)/sum(qs)))+.5*abs(math.log2(len(rq)/len(qs)))
                    +abs(math.log2((1024+mean)/(1024+mean_c))))
        nearest=sorted((r for r in self.prefill_train if bool(any(r['cached']))==bool(any(cs))),key=distance)[:3]
        if not nearest or distance(nearest[0])>1.5:return None
        weights=[1/(.1+distance(r))**2 for r in nearest]
        residual=sum(w*(r['ms']-sum(a*b for a,b in zip(r['features'],self.prefill['coefficients'])))
                     for w,r in zip(weights,nearest))/sum(weights)
        return max(1.,base+residual)*self.prefill_correction

    def decode_reserve(self, seqs, config, capable):
        if not seqs:return 0.
        features=self.features(seqs,config,capable)
        estimate=self.estimate(features)
        if estimate is not None:return estimate
        # Conservative ordinary-path upper shape for an uncalibrated small batch.
        candidates=[r for r in self.rows if not r['shared'] and r['batch']>=len(seqs)
                    and r['common']==features['common'] and r['context']>=features['context']*.95]
        return min((r['ms'] for r in candidates),default=None)

    def observe_prefill_prediction(self, prediction, elapsed_ms):
        if prediction and prediction>0:
            ratio=max(.8,min(1.5,elapsed_ms/prediction))
            self.prefill_correction=max(.8,min(1.5,self.prefill_correction*(.95+.05*ratio)))
