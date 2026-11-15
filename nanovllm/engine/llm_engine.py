import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.scheduler.shared_capable = self.model_runner.shared_supported and (
            config.shared_attention_backend!='auto' or self.model_runner.shared_auto_calibrated)
        self.config = config
        self._in_step = False
        self.scheduler.block_manager.copy_blocks = self.model_runner.copy_kv_blocks
        if config.cpu_cache_pages:
            from nanovllm.engine.transfer_manager import TransferManager
            self.transfer_manager = TransferManager(self.scheduler.block_manager,
                self.model_runner.kv_cache,config,self.scheduler.cost)
        atexit.register(self.exit)

    def exit(self):
        if not hasattr(self, "model_runner"):
            return
        if self._in_step:
            raise RuntimeError('cannot exit during a step')
        for request_id in list(self.scheduler.requests):
            self.scheduler.cancel(request_id)
        for session_id in list(self.scheduler.sessions):
            self.scheduler.close_session(session_id)
        if hasattr(self, 'transfer_manager'):
            self.transfer_manager.close()
            del self.transfer_manager
        self.scheduler.block_manager.copy_blocks = None
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def _boundary(self):
        if self._in_step:
            raise RuntimeError('session operations require a step boundary')
        if self.config.tensor_parallel_size != 1:
            raise NotImplementedError('session branching currently supports single-GPU engines')

    def _validate_session_target(self, session_id):
        if session_id is not None:
            hash(session_id)
            if any(getattr(s, 'save_session_id', None) == session_id for s in self.scheduler.requests.values()):
                raise ValueError('another active request is writing this session')

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams, session_id=None):
        if self._in_step:
            raise RuntimeError('request mutation during step')
        if session_id is not None:
            self._boundary()
        self._validate_session_target(session_id)
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not prompt or len(prompt) + sampling_params.max_tokens > self.config.max_model_len:
            raise ValueError('empty prompt or request exceeds max_model_len')
        if sampling_params.max_tokens < 1:
            raise ValueError('max_tokens must be positive')
        seq = Sequence(prompt, sampling_params)
        seq.save_session_id = session_id
        self.scheduler.add(seq)
        return seq.seq_id

    def save_session(self, request_id, session_id):
        self._boundary()
        hash(session_id)
        self.scheduler.save_session(request_id, session_id)

    def close_session(self, session_id):
        self._boundary()
        return self.scheduler.close_session(session_id)

    def pause_request(self, request_id):
        self._boundary()
        self.scheduler.pause(request_id)

    def resume_request(self, request_id):
        self._boundary()
        self.scheduler.resume(request_id)

    def cancel_request(self, request_id):
        self._boundary()
        return self.scheduler.cancel(request_id)

    def _fork(self, source, count, sampling_params, append_token_ids, next_token_ids, cache_mode, session_id):
        self._boundary()
        self._validate_session_target(session_id)
        if count < 1 or (count > 1 and session_id is not None):
            raise ValueError('invalid fork count or shared output session')
        if next_token_ids is not None and len(next_token_ids) != count:
            raise ValueError('one pending token override is required per branch')
        sampling_params = sampling_params or SamplingParams()
        if sampling_params.max_tokens < 1:
            raise ValueError('max_tokens must be positive')
        targets = []
        vocab_size = self.config.hf_config.vocab_size
        for i in range(count):
            tokens = list(source.token_ids)
            if next_token_ids is not None:
                tokens[-1] = int(next_token_ids[i])
            tokens.extend(append_token_ids or [])
            if len(tokens) + sampling_params.max_tokens > self.config.max_model_len:
                raise ValueError('branch exceeds max_model_len')
            if any(t < 0 or t >= vocab_size for t in tokens):
                raise ValueError('token outside vocabulary')
            seq = Sequence(tokens, sampling_params)
            seq.save_session_id = session_id
            targets.append(seq)
        self.scheduler.check_admission(len(targets))
        self.scheduler.block_manager.clone_many(source, targets, cache_mode)
        for seq in targets:
            self.scheduler.register(seq)
            if len(seq) - seq.num_cached_tokens == 1:
                seq.status = SequenceStatus.RUNNING
                seq.is_prefill = False
                self.scheduler.running.append(seq)
            else:
                self.scheduler.waiting.append(seq)
        return [s.seq_id for s in targets]

    def fork_requests(self, request_id, count=1, sampling_params=None, *, append_token_ids=None,
                      next_token_ids=None, cache_mode='cow'):
        return self._fork(self.scheduler.requests[request_id], count, sampling_params,
                          append_token_ids, next_token_ids, cache_mode, None)

    def fork_request(self, request_id, sampling_params=None, *, append_token_ids=None,
                     next_token_id=None, cache_mode='cow', session_id=None):
        return self._fork(self.scheduler.requests[request_id], 1, sampling_params, append_token_ids,
                          None if next_token_id is None else [next_token_id], cache_mode, session_id)[0]

    def fork_session(self, session_id, sampling_params=None, *, append_token_ids=None,
                     next_token_id=None, cache_mode='cow', output_session_id=None):
        return self._fork(self.scheduler.sessions[session_id], 1, sampling_params, append_token_ids,
                          None if next_token_id is None else [next_token_id], cache_mode, output_session_id)[0]

    def cache_stats(self):
        return self.scheduler.block_manager.stats()

    def step(self):
        if self._in_step:
            raise RuntimeError('reentrant step')
        self._in_step = True
        step_started = perf_counter()
        try:
            try:
                seqs, is_prefill = self.scheduler.schedule()
            except BaseException:
                for seq in self.scheduler.requests.values():
                    if seq.num_scheduled_tokens:
                        seq.num_scheduled_tokens = 0
                        if seq.is_prefill:
                            self.scheduler._remove_queues(seq)
                            seq.status = SequenceStatus.WAITING
                            self.scheduler.waiting.appendleft(seq)
                raise
            num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
            try:
                token_ids = self.model_runner.call('run', seqs, is_prefill)
            except BaseException:
                # Potentially partially written pages are discarded before retry.
                for seq in seqs:
                    self.scheduler._remove_queues(seq)
                    self.scheduler.preempt(seq)
                raise
            self.scheduler.record_execution(seqs, is_prefill, (perf_counter()-step_started)*1000)
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
            return outputs, num_tokens
        finally:
            self._in_step = False

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
