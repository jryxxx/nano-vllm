import sys,runpy,torch
mode=sys.argv[1]
if mode=="cache128":torch._dynamo.config.cache_size_limit=128
elif mode=="eager":
 from nanovllm.layers.layernorm import RMSNorm
 RMSNorm.rms_forward=RMSNorm.rms_forward._torchdynamo_orig_callable
 RMSNorm.add_rms_forward=RMSNorm.add_rms_forward._torchdynamo_orig_callable
sys.argv=["experiments/bench_cache.py","--mode","validate","--output",f"results/finalization/norm-{mode}.json"]
runpy.run_path("experiments/bench_cache.py",run_name="__main__")
