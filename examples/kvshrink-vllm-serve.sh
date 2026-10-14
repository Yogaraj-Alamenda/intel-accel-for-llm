#!/bin/bash -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../setvars.sh"
: "${MODEL:?Set MODEL in setvars.sh or the environment}"

export LD_PRELOAD="/usr/local/lib/libiomp5.so${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4${LD_PRELOAD:+:$LD_PRELOAD}"

parallel_args=(-tp "$TP_SIZE")
if [[ "${DP_SIZE:-1}" -gt 1 ]]; then
    parallel_args+=(--data-parallel-size "$DP_SIZE")
fi

# vLLM CPU inference sizes its KV cache from VLLM_CPU_KVCACHE_SPACE (GiB) instead.
device_args=(--gpu-memory-utilization 0.8)
[[ "$DEVICE" == cpu ]] && device_args=()

# KVSHRINK_CONNECTOR=0 serves plain vLLM with its own prefix cache, as a baseline.
connector_args=(
    --kv-transfer-config '{"kv_connector":"KVShrinkConnector","kv_connector_module_path":"kvshrink.kvshrink_connector","kv_role":"kv_both"}'
    --no-enable-prefix-caching
)
[[ "${KVSHRINK_CONNECTOR:-1}" == 0 ]] && connector_args=()

# Extra arguments are passed on to `vllm serve`.
vllm serve "$MODEL" \
    "${connector_args[@]}" \
    --trust-remote-code \
    "${parallel_args[@]}" \
    "${device_args[@]}" \
    --max-model-len "${MAX_MODEL_LEN:-32765}" \
    --block-size "${BLOCK_SIZE:-16}" \
    --port "${PORT:-8000}" \
    "$@" \
    2>&1 | tee log.kvshrink-vllm
