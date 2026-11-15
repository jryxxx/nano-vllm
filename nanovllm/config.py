import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    shared_attention_backend: str = "off"
    shared_min_batch: int = 32
    shared_min_prefix: int = 1024
    schedule_policy: str = "baseline"
    schedule_prefill_chunk: int = 1024
    schedule_prefill_mode: str = "fixed"
    schedule_phase_mode: str = "guarded"
    schedule_service_budget_ms: float = 70.0
    schedule_prediction_margin: float = 1.10
    schedule_planning_margin_ms: float = 2.0
    schedule_decode_interval_ms: float = 25.0
    schedule_wait_ms: float = 100.0
    schedule_max_decode_steps: int = 4
    schedule_candidate_limit: int = 512
    schedule_max_groups: int = 4
    schedule_queue_limit: int = 0
    schedule_cost_profile: str = ""
    cache_policy: str = "fifo"
    cache_half_life: float = 32.0
    cpu_cache_pages: int = 0
    cpu_cache_staging: int = 2
    cpu_cache_restore: str = "auto"
    cpu_cache_restore_budget_ms: float = 20.0
    max_kvcache_blocks: int = 0
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert self.cpu_cache_pages >= 0 and self.cpu_cache_staging > 0
        assert self.cpu_cache_restore in ('auto','always')
        assert self.cpu_cache_restore_budget_ms > 0 and self.max_kvcache_blocks >= 0
        assert not self.cpu_cache_pages or self.tensor_parallel_size == 1
        assert self.cache_policy in ('fifo','lru','leaf_lru','cost')
        assert self.cache_half_life > 0
        assert self.schedule_policy in ('baseline','fcfs','lpm','cost')
        assert self.schedule_prefill_chunk > 0
        assert self.schedule_prefill_mode in ('fixed','model')
        assert self.schedule_phase_mode in ('guarded','predictive')
        assert self.schedule_service_budget_ms > 0 and self.schedule_prediction_margin >= 1
        assert self.schedule_planning_margin_ms >= 0
        assert self.schedule_decode_interval_ms > 0 and self.schedule_wait_ms > 0
        assert self.schedule_max_decode_steps > 0 and self.schedule_candidate_limit > 0
        assert self.schedule_max_groups > 0 and self.schedule_queue_limit >= 0
        assert self.shared_attention_backend in ("off", "auto", "force")
        assert self.shared_min_batch >= 1 and self.shared_min_prefix >= 0
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
