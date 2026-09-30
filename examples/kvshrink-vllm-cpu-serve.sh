#!/usr/bin/env bash
set -euo pipefail

export DEVICE=cpu
export MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
export TP_SIZE="${TP_SIZE:-1}"
export VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-4}"

# --- Core placement -------------------------------------------------------------------------
# Measured: the codec thread must not share a core with inference (a floating poller loses
# 13-21% restore throughput; two pollers on inference cores halve throughput and triple TPOT).
# Default: each rank gets an equal contiguous share of the visible CPUs; the last IAXL_CORES of
# each share run that rank's IAXL threads and the rest run its inference threads.
IAXL_CORES="${IAXL_CORES:-2}"
if [[ -z "${VLLM_CPU_OMP_THREADS_BIND:-}" || -z "${KVSHRINK_CPU_OMP_THREADS_BIND:-}" ]]; then
    PLAN=$(python3 - "$TP_SIZE" "$IAXL_CORES" <<'PY'
import os, sys

tp, iaxl = int(sys.argv[1]), int(sys.argv[2])
cpus = sorted(os.sched_getaffinity(0))
share = len(cpus) // tp
if share <= iaxl:
    sys.exit(f"need more than {iaxl} CPUs per rank, have {share}")

def fmt(ids):
    runs = []
    for c in ids:
        if runs and c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in runs)

ranks = [cpus[r * share:(r + 1) * share] for r in range(tp)]
print("|".join(fmt(r[:-iaxl]) for r in ranks), "|".join(fmt(r[-iaxl:]) for r in ranks), share - iaxl)
PY
    )
    read -r BIND IAXL_BIND PER_RANK <<<"$PLAN"
    export VLLM_CPU_OMP_THREADS_BIND="${VLLM_CPU_OMP_THREADS_BIND:-$BIND}"
    export KVSHRINK_CPU_OMP_THREADS_BIND="${KVSHRINK_CPU_OMP_THREADS_BIND:-$IAXL_BIND}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$PER_RANK}"
fi
# OMP_NUM_THREADS must match the per-rank bind width when both are given explicitly.
: "${OMP_NUM_THREADS:?set OMP_NUM_THREADS to the CPUs per rank in VLLM_CPU_OMP_THREADS_BIND}"
# IAXL's own thread pool keeps it off vLLM's OpenMP runtime and lets idle pollers sleep.
export IAXL_USE_OMP="${IAXL_USE_OMP:-0}"

# --- Codec ----------------------------------------------------------------------------------
export IAXL_KV_COMPRESSION="${IAXL_KV_COMPRESSION:-1}"
export IAXL_QAT_ZIP_ENABLE="${IAXL_QAT_ZIP_ENABLE:-1}"
export IAXL_CPU_ZIP_ENABLE="${IAXL_CPU_ZIP_ENABLE:-0}"
export IAXL_IAA_ZIP_ENABLE="${IAXL_IAA_ZIP_ENABLE:-0}"
# bf16 byte-plane transform. Off (default): 19% of KV bytes saved.
# On: 28% saved, at the cost of a CPU byte-plane pass over every block.
export IAXL_KV_DATA_SHUFFLE="${IAXL_KV_DATA_SHUFFLE:-0}"
export IAXL_KV_LOSSY_TRUNC="${IAXL_KV_LOSSY_TRUNC:-0}"
export IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS="${IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS:-0}"

# One poller thread per QAT device (IAXL_OMP_THREAD_NUM) sweeps all of that device's instances.
export IAXL_IAA_INSTANCE_NUM="${IAXL_IAA_INSTANCE_NUM:-1}"

# --- Cache ----------------------------------------------------------------------------------
export IAXL_DDR_POOL_SIZE_GB="${IAXL_DDR_POOL_SIZE_GB:-4}"
export PYTHONOPTIMIZE=0

# Async layer loading was not part of the CPU benchmark campaign; leave it off until measured.
export KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED="${KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED:-0}"
export KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS="${KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS:--1}"
export KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC="${KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC:-0}"
export KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP="${KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP:-0-:0}"

echo "[launch] inference: OMP_NUM_THREADS=$OMP_NUM_THREADS bind=$VLLM_CPU_OMP_THREADS_BIND | iaxl: bind=$KVSHRINK_CPU_OMP_THREADS_BIND use_omp=$IAXL_USE_OMP qat_devices=${KVSHRINK_QAT_DEVICES:-${IAXL_QAT_DEVICES:-0}} shuffle=$IAXL_KV_DATA_SHUFFLE" >&2

python - <<'PY'
from iaxl import torch_ext
from vllm.platforms import current_platform

if torch_ext.device_type != "cpu":
    raise SystemExit("Build IAXL with DEVICE=cpu before starting CPU inference.")
if not current_platform.is_cpu():
    raise SystemExit("Use a CPU-enabled vLLM and PyTorch installation.")
PY

# KVSHRINK_CONNECTOR=0 serves plain vLLM with its own prefix cache (the baseline arm).
if [[ "${KVSHRINK_CONNECTOR:-1}" == "1" ]]; then
    KV_ARGS=(--kv-transfer-config '{"kv_connector":"KVShrinkConnector","kv_connector_module_path":"kvshrink.kvshrink_connector","kv_role":"kv_both"}'
             --no-enable-prefix-caching)
else
    KV_ARGS=(--enable-prefix-caching)
fi

exec vllm serve "$MODEL" \
    "${KV_ARGS[@]}" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype "${DTYPE:-bfloat16}" \
    --enforce-eager \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    --max-num-seqs "${MAX_NUM_SEQS:-8}" \
    --block-size "${BLOCK_SIZE:-32}" \
    --port "${PORT:-8000}" \
    "$@"