# vLLM CPU Inference with KVShrink

[简体中文](vllm-cpu-inference.zh-CN.md)

This guide shows how to configure threads and CPU binding when vLLM runs inference on the CPU and KVShrink compresses the KV cache on Intel QAT or IAA. It requires a CPU build of vLLM (`VLLM_TARGET_DEVICE=cpu`) in the container and IAXL built with `DEVICE=cpu`. For the design behind these settings, see the [KV Zip design](../design/kv-zip.md).

## 1. Build

A `DEVICE=cpu` build uses the same `KVStore`, codecs, and persist/evict APIs as GPU inference. It needs no CUDA, SYCL, GDRCopy, or NIXL, so DSA + GDRCopy transfers and RDMA (`remote_pool`) are not available; Intel DSA can still take over host-to-host block copies (see [Cache Settings](#4-cache-settings)). `docker/Dockerfile.dev` builds GPU dependencies (UCX, NIXL), so the build may also be run directly in an environment that has CPU vLLM installed. Besides the Python build requirements, it needs a C/C++ compiler, CMake, NASM (for the QPL dependency, even with IAA disabled), SQLite, and zlib.

```bash
python -m pip install -r requirements.txt
DEVICE=cpu python -m pip install -e . --no-build-isolation                     # out-of-tree QAT driver package
DEVICE=cpu IAXL_QATLIB=intree python -m pip install -e . --no-build-isolation # in-tree driver with system qatlib
```

The out-of-tree build also needs the Boost/Boost Regex, SSL, udev, and netlink development packages. Match the user-space library to the host driver: the out-of-tree runtime expects `/dev/qat_dev_processes`, while qatlib uses the host's VFIO setup.

## 2. CPU Binding

Two variables split each rank's CPUs between vLLM and IAXL:

| Variable | Used by | Threads |
| --- | --- | --- |
| `VLLM_CPU_OMP_THREADS_BIND` | vLLM | OpenMP inference threads, one per listed CPU |
| `KVSHRINK_CPU_OMP_THREADS_BIND` | IAXL | `OMP-Main`, `POOL-<n>`, `H2D`, `D2H`, and other IAXL threads |

- **Format:** both variables contain one entry per rank, separated by `|`. Each entry is a CPU list such as `0-61` or `0-7,16`.
- **Rank order:** entry `r` belongs to global rank `r = dp_rank × TP_SIZE + tp_rank`, so each list needs `TP_SIZE × DP_SIZE` entries.
- **Default:** `KVSHRINK_CPU_OMP_THREADS_BIND` defaults to `VLLM_CPU_OMP_THREADS_BIND`. In that case IAXL threads share the inference cores, so set it explicitly for CPU inference.
- **At startup:** the connector exports the rank's entry as `IAXL_CPU_OMP_THREADS_BIND`, and IAXL pins all of its threads to those CPUs.

When choosing CPUs:

- For each rank, keep the vLLM and IAXL entries disjoint.
- Put both entries on the same NUMA node. Choose the rank's QAT devices (`KVSHRINK_QAT_DEVICES`) on that node too.
- Use one logical CPU per physical core. `lscpu -e=CPU,CORE,NODE` shows the mapping.
- Give IAXL at least `IAXL_OMP_THREAD_NUM` CPUs per rank.

## 3. Thread Settings

| Variable | Recommended | Purpose |
| --- | --- | --- |
| `IAXL_USE_OMP` | `0` (default with `DEVICE=cpu`) | IAXL runs on its own thread pool instead of OpenMP, which leaves OpenMP to vLLM. `setvars.sh` then unsets `OMP_NUM_THREADS` and `OMP_THREAD_LIMIT`. |
| `IAXL_OMP_THREAD_NUM` | `1` or `2` | Number of threads driving the compression workers. One thread can keep several QAT/IAA instances busy. Valid range: 1 to the worker count. |
| `IAXL_CPU_ZIP_ENABLE` | `0` | Spends no cores on zlib; QAT and IAA do the compression. |
| `IAXL_QAT_ZIP_ENABLE`, `IAXL_IAA_ZIP_ENABLE` | `1` for the accelerators you have | The worker count is `IAXL_QAT_INSTANCE_NUM + IAXL_IAA_INSTANCE_NUM`; `setvars.sh` prints both values. |

## 4. Cache Settings

`examples/kvshrink-vllm-serve.sh` drops `--gpu-memory-utilization` when `DEVICE=cpu`. These variables commonly need overriding for CPU inference:

| Variable | Purpose |
| --- | --- |
| `VLLM_CPU_KVCACHE_SPACE` | Live vLLM KV cache size in GiB. `IAXL_DDR_POOL_SIZE_GB` sizes the compressed KVShrink cache independently. |
| `MAX_MODEL_LEN` | `--max-model-len` (default `32765`). Lower it when the CPU KV cache cannot hold one full-length request. |
| `BLOCK_SIZE` | `--block-size` (default `16`). A CPU KV block holds `2 × num_kv_heads × block_size × head_size × dtype_bytes` bytes per layer and must fit in `IAXL_ZIP_SRC_CAP` and, compressed, in `IAXL_ZIP_DST_CAP` (both 256 KiB by default). |
| `IAXL_KV_DATA_SHUFFLE`, `IAXL_KV_LOSSY_TRUNC` | The codec reads blocks straight from the vLLM KV tensor and writes them straight back, so neither setting adds a staging copy. `setvars.sh` enables shuffle by default: it improves the compression ratio, and for BF16 it is applied while the block is copied into the codec buffer and undone while the output is copied back. |
| `IAXL_DSA_MEMCPY_ENABLE` | `1` moves plain block copies to Intel DSA (off by default): raw (uncompressed) blocks in both directions, including layers that bypass compression, as one DSA batch, and the per-block QAT/IAA copies (compressed payload into the device buffer, gather into it, scatter back into the KV tensor), which run asynchronously next to the codec. Byte-shuffled blocks are still transformed on the CPU. Needs a user-mode DSA work queue (`/dev/dsa/wqX.Y`, see `tools/setup_dsa.sh`); on the first DSA failure the process warns once and uses `memcpy` from then on. |
| `IAXL_DSA_WQS`, `KVSHRINK_DSA_DEVICES` | DSA work queues for the process, or one `\|`-separated entry per rank (for example `wq0.0\|wq2.0`), which the connector exports as each rank's `IAXL_DSA_WQS`. |
| `IAXL_DSA_MEMCPY_MIN_BYTES` | Raw-block batches smaller than this (default 1 MiB) stay on `memcpy`, where the DSA round trip would cost more than the copy. |

## 5. Example

This example runs two ranks on a host where NUMA node 0 has CPUs 0-63 and node 1 has CPUs 64-127, with one logical CPU per core:

```bash
export DEVICE=cpu # start.sh passes it into the container, which builds IAXL on entry
./start.sh

# In the container shell opened by start.sh
export TP_SIZE=2
export VLLM_CPU_OMP_THREADS_BIND="0-61|64-125"       # vLLM inference
export KVSHRINK_CPU_OMP_THREADS_BIND="62-63|126-127" # IAXL threads
export KVSHRINK_QAT_DEVICES="0,1|2,3"                # QAT devices on each rank's NUMA node
export IAXL_CPU_ZIP_ENABLE=0
export IAXL_OMP_THREAD_NUM=2

./examples/kvshrink-vllm-serve.sh
```

The script sources `setvars.sh`. It also preloads libiomp5 and libtcmalloc, as vLLM's CPU backend recommends.

## 6. Verification

- **`setvars.sh` output:** check `VLLM_CPU_OMP_THREADS_BIND`, `KVSHRINK_CPU_OMP_THREADS_BIND`, `IAXL_OMP_THREAD_NUM`, and `IAXL_USE_OMP=0`.
- **vLLM log:** each rank prints `Bound rank <r>: IAXL_CPU_OMP_THREADS_BIND=<cpus>` and `[iaxl] config: ... omp_threads=2 use_omp=OFF ...`. With `IAXL_DSA_MEMCPY_ENABLE=1` the config line shows `dsa_memcpy=ON`, and the first DSA copies log `[kv_zip] segment copies: using Intel DSA` or `[kv_zip] codec staging copies: using Intel DSA`.
- **Thread affinity:** after the first request, run the command below. IAXL threads (`OMP-Main`, `POOL-*`, `H2D`, `D2H`) should only list the rank's `KVSHRINK_CPU_OMP_THREADS_BIND` entry.

  ```bash
  for f in $(grep -l "^OMP-Main$" /proc/[0-9]*/task/*/comm); do
      pid=$(cut -d/ -f3 <<<"$f"); echo "== pid $pid"
      for t in /proc/$pid/task/*; do
          echo "$(cat $t/comm) $(grep Cpus_allowed_list $t/status | cut -f2)"
      done | sort | uniq -c
  done
  ```

## 7. Troubleshooting

| Message | Fix |
| --- | --- |
| `KVSHRINK_CPU_OMP_THREADS_BIND has N entries, but rank is R` | Provide one entry per rank (`TP_SIZE × DP_SIZE`). |
| `invalid IAXL_CPU_OMP_THREADS_BIND` | An entry is not a valid CPU list. |
| `cannot bind to IAXL_CPU_OMP_THREADS_BIND` | None of the listed CPUs is available to the process, for example because they are outside the container's cpuset. |
| `IAXL_OMP_THREAD_NUM must be between 1 and ...` | Lower `IAXL_OMP_THREAD_NUM`, or enable more QAT/IAA instances. |
| `IAXL_OMP_THREAD_NUM (...) + reserved (...) exceeds the smallest rank CPU allocation` | The smallest `VLLM_CPU_OMP_THREADS_BIND` entry is too small. Lower `IAXL_RESERVED_CPU_NUM` or `IAXL_OMP_THREAD_NUM`. |
| vLLM warns that libiomp or libtcmalloc is not in `LD_PRELOAD` | Start vLLM with `examples/kvshrink-vllm-serve.sh`. |
| `[kv_zip] WARNING: ... falling back to CPU memcpy for the rest of this process` | The DSA work queue in `IAXL_DSA_WQS` is missing, not user-mode, or failed a copy. Check `/dev/dsa/` and `tools/setup_dsa.sh`, or set `IAXL_DSA_MEMCPY_ENABLE=0`. |
| `RDMA (remote_pool) needs NIXL, which DEVICE=cpu builds do not link` | Cross-node KV cache reuse needs a CUDA or XPU build. |
