# MovieRoll on vLLM-Omni: environment defaults for several GPUs (source this file; it keeps any value already set).
#   . vllm_omni_movieroll/examples/movieroll_env.sh
# Used by examples/online_serving/movieroll/run_server.sh; offline, source it before vLLM-Omni's shared example scripts
# or a Python program that builds `Omni` (README, Parallelism and PS-VAE).

# vLLM's multi-process executor otherwise sets OMP_NUM_THREADS to every CPU of the host, which slows the engine's
# per-frame video post-processing (README, Parallelism and PS-VAE)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
# vLLM-Omni's multi-process executor stops every worker when one request's model call takes longer than this many seconds
# (600 by default); one minute of Base on 8 GPUs computes about 30 min. MovieRoll refuses a request early, naming the
# value it needs, when its own pace projects past it.
export VLLM_OMNI_DLO_DP_WAVE_TIMEOUT="${VLLM_OMNI_DLO_DP_WAVE_TIMEOUT:-14400}"
