from collections import deque, Counter
from time import perf_counter

from nanovllm.engine.schedule_cost import ScheduleCost

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager, CacheCapacityError


class RequestQueueFull(RuntimeError):
    pass


class Scheduler:

    def __init__(self, config: Config):
        self.config = config
        self.clock = perf_counter
        self.cost = ScheduleCost(config)
        self.shared_capable = False
        self.metrics = Counter()
        self.last_phase = None
        self.decode_streak = 0
        self.prefill_prediction = None
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        coefficients = self.cost.prefill['coefficients'] if self.cost.prefill else None
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size,
                                         getattr(config, 'cache_policy', 'fifo'), getattr(config, 'cache_half_life', 32.),
                                         (coefficients[2], coefficients[3]) if coefficients else None)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.requests = {}
        self.sessions = {}

    def is_finished(self):
        return not self.waiting and not self.running

    def check_admission(self, count=1):
        limit = self.config.schedule_queue_limit
        if limit and len(self.requests)+count > limit:
            self.metrics['rejected_requests'] += count
            raise RequestQueueFull('active request limit reached; retry with backpressure')

    def register(self, seq):
        now = self.clock()
        seq.scheduler_arrival = now
        seq.scheduler_service = now
        self.requests[seq.seq_id] = seq

    def add(self, seq: Sequence):
        self.check_admission()
        self.register(seq)
        self.requests[seq.seq_id] = seq
        self.waiting.append(seq)

    def _remove_queues(self, seq):
        for queue in (self.waiting, self.running):
            if seq in queue:
                queue.remove(seq)

    def cancel(self, request_id):
        seq = self.requests.pop(request_id, None)
        if seq is None:
            return False
        self._remove_queues(seq)
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.CANCELLED
        return True

    def pause(self, request_id):
        seq = self.requests[request_id]
        self.block_manager._check_source(seq)
        self._remove_queues(seq)
        seq.status = SequenceStatus.PAUSED

    def resume(self, request_id):
        seq = self.requests[request_id]
        if seq.status != SequenceStatus.PAUSED:
            raise ValueError('request is not paused')
        seq.status = SequenceStatus.RUNNING
        seq.scheduler_service = self.clock()
        self.running.append(seq)

    def save_session(self, request_id, session_id):
        snap = self.block_manager.snapshot(self.requests[request_id])
        previous = self.sessions.get(session_id)
        self.sessions[session_id] = snap
        if previous is not None:
            self.block_manager.release(previous)

    def close_session(self, session_id):
        previous = self.sessions.pop(session_id, None)
        if previous is None:
            return False
        self.block_manager.release(previous)
        return True

    def _age(self, seq, now):
        return (now-getattr(seq,'scheduler_service',now))*1000

    def _phase(self, now):
        if not self.running: return 'prefill'
        if not self.waiting: return 'decode'
        if self.last_phase == 'prefill':
            self.metrics['phase_decode_guard'] += 1
            return 'decode'
        if self.decode_streak >= self.config.schedule_max_decode_steps:
            self.metrics['phase_wait_guard'] += 1
            return 'prefill'
        decode_due = max(self._age(s,now) for s in self.running) >= self.config.schedule_decode_interval_ms
        wait_due = max((now-getattr(s,'scheduler_arrival',now))*1000 for s in self.waiting) >= self.config.schedule_wait_ms
        if decode_due:
            self.metrics['phase_decode_due'] += 1
            return 'decode'
        if wait_due:
            self.metrics['phase_wait_guard'] += 1
            return 'prefill'
        return 'decode' if decode_due else 'prefill'

    def _order_waiting(self, now):
        if not self.waiting: return
        oldest = min(self.waiting,key=lambda s:getattr(s,'scheduler_arrival',now))
        chosen = oldest
        if (self.config.schedule_policy in ('lpm','cost') and
                (now-getattr(oldest,'scheduler_arrival',now))*1000 < self.config.schedule_wait_ms):
            candidates=list(self.waiting)[:min(16,self.config.schedule_candidate_limit)]
            def hit(s):
                return s.num_cached_tokens if s.block_table else max(0,self.block_manager.can_allocate(s))*self.block_size
            chosen=max(candidates,key=hit)
        self.waiting.remove(chosen); self.waiting.appendleft(chosen)

    def _order_decode(self, now):
        queue=list(self.running)
        fair=sorted(queue,key=lambda s:getattr(s,'scheduler_service',now))
        if getattr(self,'_refresh_count',0):
            chosen=fair[:self._refresh_count]
            selected={s.seq_id for s in chosen}
            self._decode_limit=len(chosen)
            self.running=deque(chosen+[s for s in fair if s.seq_id not in selected])
            self.metrics['decode_tail_refresh_steps']+=1
            self.metrics['decode_tail_refresh_requests']+=len(chosen)
            return
        base=fair[:self.max_num_seqs]; chosen=base
        policy=self.config.schedule_policy
        if policy in ('lpm','cost') and self._age(fair[0],now) < self.config.schedule_decode_interval_ms:
            groups={}; width=max(1,self.config.shared_min_prefix//self.block_size)
            for seq in fair[:self.config.schedule_candidate_limit]:
                if seq.num_cached_tokens < width*self.block_size: continue
                key=tuple(seq.block_table[:width])
                if key not in groups and len(groups)>=self.config.schedule_max_groups: continue
                groups.setdefault(key,[]).append(seq)
            candidates=[g[:self.max_num_seqs] for g in groups.values() if len(g)>=self.config.shared_min_batch]
            if policy=='lpm' and candidates:
                chosen=max(candidates,key=lambda g:self.cost.features(g,self.config,self.shared_capable)['prefix_tokens']*len(g))
            elif policy=='cost':
                base_ms=self.cost.estimate(self.cost.features(base,self.config,self.shared_capable))
                best_rate=len(base)/base_ms if base_ms else None
                for group in candidates:
                    ms=self.cost.estimate(self.cost.features(group,self.config,self.shared_capable))
                    if ms is None or best_rate is None: continue
                    selected={s.seq_id for s in group}
                    deferred=[s for s in fair if s.seq_id not in selected]
                    if deferred and max(self._age(s,now) for s in deferred)+ms >= self.config.schedule_decode_interval_ms:
                        continue
                    if len(group)/ms > best_rate*1.05:
                        chosen=group; best_rate=len(group)/ms
                if chosen is not base: self.metrics['cost_group_selected'] += 1
                else: self.metrics['cost_fcfs_selected'] += 1
        elif policy in ('lpm','cost'):
            self.metrics['decode_age_guard'] += 1
        selected={s.seq_id for s in chosen}
        # Restrict this step to the chosen group; leave others for fair rotation.
        self._decode_limit=len(chosen)
        self.running=deque(chosen+[s for s in fair if s.seq_id not in selected])

    def _preview_prefill(self, budget, cache):
        qs=[];cs=[]
        for seq in list(self.waiting)[:min(self.max_num_seqs,8)]:
            if budget<=0:break
            if seq.block_table:
                cached=seq.num_cached_tokens
            else:
                if seq.seq_id not in cache:cache[seq.seq_id]=self.block_manager.can_allocate(seq)
                blocks=cache[seq.seq_id]
                if blocks<0:break
                cached=blocks*self.block_size
            remaining=len(seq)-cached
            if qs and remaining>budget:break
            count=min(remaining,budget)
            if not self.block_manager.can_write(seq,count) and seq.block_table:break
            qs.append(count);cs.append(cached);budget-=count
        return qs,cs

    def _prefill_options(self, now):
        options=[];seen=set();cache={}
        running=sorted(self.running,key=lambda s:getattr(s,'scheduler_service',now))
        # Forecast the completion of oldest-first decode rounds after prefill.
        elapsed=0.;slack=self.config.schedule_service_budget_ms
        for i in range(0,len(running),self.max_num_seqs):
            group=running[i:i+self.max_num_seqs]
            ms=self.cost.decode_reserve(group,self.config,self.shared_capable)
            if ms is None:
                self.metrics['prefill_unknown_decode_cost']+=1
                return []
            elapsed+=ms*self.config.schedule_prediction_margin
            slack=min(slack,self.config.schedule_service_budget_ms-max(self._age(s,now) for s in group)-elapsed)
        for budget in (256,512,1024,2048,3072,4096,6144,8192):
            budget=min(budget,self.max_num_batched_tokens)
            qs,cs=self._preview_prefill(budget,cache)
            key=(tuple(qs),tuple(cs))
            if not qs or key in seen:continue
            seen.add(key);prediction=self.cost.predict_prefill(qs,cs)
            if prediction is None:continue
            safe=(not running or prediction*self.config.schedule_prediction_margin+
                  self.config.schedule_planning_margin_ms<=slack)
            options.append(dict(tokens=sum(qs),prediction=prediction,safe=safe))
        return options

    def schedule(self) -> tuple[list[Sequence], bool]:
        fair=self.config.schedule_policy!='baseline'
        now=self.clock();self.prefill_prediction=None;self._refresh_count=0
        if fair:self._order_waiting(now)
        phase=self._phase(now) if fair else 'prefill'
        self._step_budget=min(self.max_num_batched_tokens,self.config.schedule_prefill_chunk) if fair else self.max_num_batched_tokens
        if fair and getattr(self.config,'schedule_prefill_mode','fixed')=='model' and self.waiting:
            options=self._prefill_options(now)
            safe=[o for o in options if o['safe']]
            chosen=max(safe,key=lambda o:o['tokens']) if safe else None
            predictive=self.config.schedule_phase_mode=='predictive'
            if chosen:
                self._step_budget=chosen['tokens'];self.prefill_prediction=chosen['prediction']
                if predictive and self.last_phase!='prefill':phase='prefill'
            elif options:
                self.metrics['prefill_no_slack']+=1
                if predictive and self.running and self.decode_streak<self.config.schedule_max_decode_steps:
                    phase='decode'
                    latest=max(getattr(s,'scheduler_service',now) for s in self.running)
                    older=sum(getattr(s,'scheduler_service',now)<latest-1e-9 for s in self.running)
                    if 0<older<self.max_num_seqs:self._refresh_count=older
                else:
                    chosen=min(options,key=lambda o:o['prediction'])
                    self._step_budget=chosen['tokens'];self.prefill_prediction=chosen['prediction']
                    self.metrics['prefill_forced_progress']+=1
            else:
                self.metrics['prefill_model_fallback']+=1
        if phase=='prefill':
            result=self._prefill()
            if result:
                if self.prefill_prediction is not None:
                    # Actual allocation may shorten a preview under memory pressure.
                    self.prefill_prediction=self.cost.predict_prefill(
                        [s.num_scheduled_tokens for s in result],[s.num_cached_tokens for s in result])
                    self.metrics['prefill_model_steps']+=1
                return result,True
        self.prefill_prediction=None
        if self.running:
            if fair:self._order_decode(self.clock())
            return self._decode(fair),False
        if getattr(self.block_manager, 'transfer', None) is not None and self.block_manager.transfer.tickets:
            self.block_manager.transfer.wait()
            result = self._prefill()
            if result:return result, True
        raise CacheCapacityError('no runnable request fits; release retained cache or increase capacity')

    def _prefill(self):
        scheduled_seqs = []
        num_batched_tokens = 0
        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self._step_budget - num_batched_tokens
            if remaining == 0:
                break
            if getattr(self.block_manager, 'transfer', None) is not None and not self.block_manager.transfer.prepare(seq):
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            count = min(num_tokens, remaining)
            if not self.block_manager.can_write(seq, count):
                break
            self.block_manager.ensure_writable(seq, count)
            seq.num_scheduled_tokens = count
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        return scheduled_seqs

    def _decode(self, fair):
        scheduled_seqs=[]
        limit=self._decode_limit if fair else self.max_num_seqs
        while self.running and len(scheduled_seqs) < limit:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                try:
                    self.block_manager.may_append(seq)
                except BaseException:
                    self.running.appendleft(seq)
                    self.running.extendleft(reversed(scheduled_seqs))
                    for scheduled in scheduled_seqs:
                        scheduled.num_scheduled_tokens = 0
                    raise
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                scheduled_seqs.append(seq)
        if not scheduled_seqs:
            raise CacheCapacityError("no runnable request fits; close sessions, cancel requests, or increase KV capacity")
        if fair:
            self.running.extend(scheduled_seqs)
        else:
            self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.num_scheduled_tokens = 0
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def record_execution(self, seqs, is_prefill, elapsed_ms):
        if self.config.schedule_policy=='baseline': return
        self.last_phase='prefill' if is_prefill else 'decode'
        self.decode_streak=0 if is_prefill else self.decode_streak+1
        self.metrics[self.last_phase+'_steps'] += 1
        if is_prefill:
            self.cost.observe_prefill(sum(s.num_scheduled_tokens for s in seqs),max(len(s) for s in seqs),elapsed_ms)
            if self.prefill_prediction is not None:
                self.cost.observe_prefill_prediction(self.prefill_prediction,elapsed_ms)
                self.metrics['prefill_prediction_abs_error_us']+=int(abs(elapsed_ms-self.prefill_prediction)*1000)
                self.metrics['prefill_prediction_count']+=1
        else:
            features=self.cost.features(seqs,self.config,self.shared_capable)
            self.cost.observe(features,elapsed_ms)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        now=self.clock()
        for seq, token_id in zip(seqs, token_ids):
            seq.scheduler_service=now
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                session_id = getattr(seq, 'save_session_id', None)
                if session_id is not None:
                    self.save_session(seq.seq_id, session_id)
                seq.status = SequenceStatus.FINISHED
                self.requests.pop(seq.seq_id)
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
