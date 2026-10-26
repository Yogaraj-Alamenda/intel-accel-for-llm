#!/bin/bash -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../setvars.sh"
: "${MODEL:?Set MODEL in setvars.sh or the environment}"

export LD_PRELOAD="$IAXL_LD_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"

parallel_args=(-tp "$TP_SIZE")
if [[ "${DP_SIZE:-1}" -gt 1 ]]; then
    parallel_args+=(--data-parallel-size "$DP_SIZE")
fi

# vLLM CPU inference sizes its KV cache from VLLM_CPU_KVCACHE_SPACE (GiB) instead.
device_args=(--gpu-memory-utilization 0.8)
[[ "$DEVICE" == cpu ]] && device_args=()
# Unset BLOCK_SIZE keeps vLLM's default: 128 on CPU (16 for MLA models), 16 or the attention backend's preference on GPU.
[[ -n "$BLOCK_SIZE" ]] && device_args+=(--block-size "$BLOCK_SIZE")

# Hybrid GDN/Mamba models need prefix caching: with it off vLLM sets
# mamba_cache_mode to 'none' and no block-aligned state is kept.
cache_args=(--no-enable-prefix-caching)
if [[ "${KVSHRINK_HYBRID:-0}" == "1" ]]; then
    cache_args=(--enable-prefix-caching)
fi

vllm serve "$MODEL" \
    --kv-transfer-config '{"kv_connector":"KVShrinkConnector","kv_connector_module_path":"kvshrink.kvshrink_connector","kv_role":"kv_both"}' \
    --trust-remote-code \
    "${parallel_args[@]}" \
    "${device_args[@]}" \
    --max-model-len "${MAX_MODEL_LEN:-32765}" \
    --port "${PORT:-8000}" \
    "${cache_args[@]}" \
    2>&1 | tee log.kvshrink-vllm
