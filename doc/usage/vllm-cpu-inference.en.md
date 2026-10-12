# vLLM CPU Inference with KVShrink

[简体中文](vllm-cpu-inference.zh-CN.md)

This guide shows how to configure threads and CPU binding when vLLM runs inference on the CPU and KVShrink compresses the KV cache on Intel QAT or IAA. It requires a CPU build of vLLM (`VLLM_TARGET_DEVICE=cpu`) in the container. For the design behind these settings, see the [KV Zip design](../design/kv-zip.md).

## 1. CPU Binding

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

## 2. Thread Settings

| Variable | Recommended | Purpose |
| --- | --- | --- |
| `IAXL_USE_OMP` | `0` | IAXL runs on its own thread pool instead of OpenMP, which leaves OpenMP to vLLM. `setvars.sh` then unsets `OMP_NUM_THREADS` and `OMP_THREAD_LIMIT`. |
| `IAXL_OMP_THREAD_NUM` | `1` or `2` | Number of threads driving the compression workers. One thread can keep several QAT/IAA instances busy. Valid range: 1 to the worker count. |
| `IAXL_CPU_ZIP_ENABLE` | `0` | Spends no cores on zlib; QAT and IAA do the compression. |
| `IAXL_QAT_ZIP_ENABLE`, `IAXL_IAA_ZIP_ENABLE` | `1` for the accelerators you have | The worker count is `IAXL_QAT_INSTANCE_NUM + IAXL_IAA_INSTANCE_NUM`; `setvars.sh` prints both values. |

## 3. Example

This example runs two ranks on a host where NUMA node 0 has CPUs 0-63 and node 1 has CPUs 64-127, with one logical CPU per core:

```bash
./start.sh

# In the container shell opened by start.sh
export TP_SIZE=2
export VLLM_CPU_OMP_THREADS_BIND="0-61|64-125"       # vLLM inference
export KVSHRINK_CPU_OMP_THREADS_BIND="62-63|126-127" # IAXL threads
export KVSHRINK_QAT_DEVICES="0,1|2,3"                # QAT devices on each rank's NUMA node
export IAXL_USE_OMP=0
export IAXL_CPU_ZIP_ENABLE=0
export IAXL_OMP_THREAD_NUM=2

./examples/kvshrink-vllm-serve.sh
```

The script sources `setvars.sh`. It also preloads libiomp5 and libtcmalloc, as vLLM's CPU backend recommends.

## 4. Verification

- **`setvars.sh` output:** check `VLLM_CPU_OMP_THREADS_BIND`, `KVSHRINK_CPU_OMP_THREADS_BIND`, `IAXL_OMP_THREAD_NUM`, and `IAXL_USE_OMP=0`.
- **vLLM log:** each rank prints `Bound rank <r>: IAXL_CPU_OMP_THREADS_BIND=<cpus>` and `[iaxl] config: ... omp_threads=2 use_omp=OFF ...`.
- **Thread affinity:** after the first request, run the command below. IAXL threads (`OMP-Main`, `POOL-*`, `H2D`, `D2H`) should only list the rank's `KVSHRINK_CPU_OMP_THREADS_BIND` entry.

  ```bash
  for f in $(grep -l "^OMP-Main$" /proc/[0-9]*/task/*/comm); do
      pid=$(cut -d/ -f3 <<<"$f"); echo "== pid $pid"
      for t in /proc/$pid/task/*; do
          echo "$(cat $t/comm) $(grep Cpus_allowed_list $t/status | cut -f2)"
      done | sort | uniq -c
  done
  ```

## 5. Troubleshooting

| Message | Fix |
| --- | --- |
| `KVSHRINK_CPU_OMP_THREADS_BIND has N entries, but rank is R` | Provide one entry per rank (`TP_SIZE × DP_SIZE`). |
| `IAXL threads (...) share N of rank R's inference CPUs with T codec threads` | IAXL CPUs overlap the rank's `VLLM_CPU_OMP_THREADS_BIND` entry, or are unset and default to it. Give IAXL disjoint CPUs, or set `IAXL_OMP_THREAD_NUM=1` (a single thread only logs a warning). |
| `invalid IAXL_CPU_OMP_THREADS_BIND` | An entry is not a valid CPU list. |
| `cannot bind to IAXL_CPU_OMP_THREADS_BIND` | None of the listed CPUs is available to the process, for example because they are outside the container's cpuset. |
| `IAXL_OMP_THREAD_NUM must be between 1 and ...` | Lower `IAXL_OMP_THREAD_NUM`, or enable more QAT/IAA instances. |
| `IAXL_OMP_THREAD_NUM (...) + reserved (...) exceeds the smallest rank CPU allocation` | The smallest `VLLM_CPU_OMP_THREADS_BIND` entry is too small. Lower `IAXL_RESERVED_CPU_NUM` or `IAXL_OMP_THREAD_NUM`. |
| vLLM warns that libiomp or libtcmalloc is not in `LD_PRELOAD` | Start vLLM with `examples/kvshrink-vllm-serve.sh`. |
