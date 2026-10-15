#!/bin/bash -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../setvars.sh"
: "${MODEL:?Set MODEL in setvars.sh or the environment}"

# libiomp5 comes from intel-openmp: /usr/local/lib on the GPU image, $VIRTUAL_ENV/lib on the CPU image.
IOMP5_LIB=/usr/local/lib/libiomp5.so
[[ -n "$VIRTUAL_ENV" && -f "$VIRTUAL_ENV/lib/libiomp5.so" ]] && IOMP5_LIB="$VIRTUAL_ENV/lib/libiomp5.so"
export LD_PRELOAD="$IOMP5_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4${LD_PRELOAD:+:$LD_PRELOAD}"

parallel_args=(-tp "$TP_SIZE")
if [[ "${DP_SIZE:-1}" -gt 1 ]]; then
    parallel_args+=(--data-parallel-size "$DP_SIZE")
fi

# vLLM CPU inference sizes its KV cache from VLLM_CPU_KVCACHE_SPACE (GiB) instead.
device_args=(--gpu-memory-utilization 0.8)
[[ "$DEVICE" == cpu ]] && device_args=()

vllm serve "$MODEL" \
    --kv-transfer-config '{"kv_connector":"KVShrinkConnector","kv_connector_module_path":"kvshrink.kvshrink_connector","kv_role":"kv_both"}' \
    --trust-remote-code \
    "${parallel_args[@]}" \
    "${device_args[@]}" \
    --max-model-len "${MAX_MODEL_LEN:-32765}" \
    --block-size "${BLOCK_SIZE:-16}" \
    --port "${PORT:-8000}" \
    --no-enable-prefix-caching \
    2>&1 | tee log.kvshrink-vllm
