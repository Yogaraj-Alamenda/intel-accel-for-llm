[中文](README.zh-CN.md)

# intel-accel-for-llm (`iaxl`)

`iaxl` uses Intel hardware accelerators to improve LLM inference performance.

## Design Documentation

- [IAXL Design](doc/design/iaxl.md)
- [KV Zip Design](doc/design/kv-zip.md)
- [KVShrink Design](doc/design/kvshrink.md)

## Host Setup

1. Add `intel_iommu=on,sm_on iommu=pt` to the kernel command line, then reboot the host:

```bash
sudo ./tools/setup_kernel_cmdline.sh
sudo reboot
```

2. After rebooting, install a QAT driver. Use **one** of the following.

   **Out-of-tree driver package** (default):

```bash
wget -q https://downloadmirror.intel.com/843052/QAT20.L.1.2.30-00078.tar.gz
tar xf QAT20.L.1.2.30-00078.tar.gz
./configure
make -j$(nproc)
sudo make install
```

Use the following commands to stop or start the out-of-tree QAT service:

```bash
adf_ctl down
adf_ctl up
```

   **In-tree driver + qatlib** (kernel `qat_4xxx` module, virtual functions bound to `vfio-pci`):

```bash
sudo apt install qatlib-service libqat-dev libusdm-dev   # or build https://github.com/intel/qatlib
printf 'POLICY=0\nServicesEnabled=dc\n' | sudo tee /etc/sysconfig/qat
sudo systemctl enable qat
sudo systemctl restart qat
export IAXL_QATLIB=intree   # before sourcing setvars.sh / building
```

`POLICY=0` gives every process one virtual function from each QAT device, which `IAXL_QAT_DEVICES` / `KVSHRINK_QAT_DEVICES` index. With `POLICY=1` each process only sees device `0`.

With `IAXL_QATLIB=intree` the build links the system `libqat` / `libusdm` found by `pkg-config qatlib` instead of downloading and building the out-of-tree package into `_lib`. For container builds, the qatlib development packages must be installed inside the container, and `/run/qat` must be mounted from the host at runtime.

3. Install the GDRCopy driver and configure DSA:

```bash
sudo ./tools/install_gdr_driver.sh
./tools/setup_dsa_cnt.sh
```

## Environment Variables

Common settings in `setvars.sh`. Other variables in `setvars.sh` are advanced settings that most users do not need to change.

| Category | Environment variable | Default | Description |
| --- | --- | --- | --- |
| Accelerators | `IAXL_QAT_ZIP_ENABLE` | `1` | Enable QAT compression workers (`0`/`1`) |
| Accelerators | `IAXL_IAA_ZIP_ENABLE` | `0` | Enable Intel IAA compression workers via Intel QPL (`0`/`1`). Can be combined with QAT; each block records whether IAA can decode it |
| Accelerators | `IAXL_DSA_GD_ENABLE` | `0` | Enable Intel DSA + GDRCopy transfers (`0`/`1`) |
| Compression | `IAXL_KV_COMPRESSION` | `1` | Enable DEFLATE compression of the KV cache (`0`/`1`) |
| Compression | `IAXL_KV_DATA_SHUFFLE` | `1` | Byte-shuffle BF16 data before compression (`0`/`1`). Slightly lowers compression throughput, but improves the compression ratio and decompression throughput |
| vLLM | `MODEL` | `Qwen/Qwen3-32B` | Hugging Face model ID or local model path |
| vLLM | `TP_SIZE` | `2` | Required; number of Tensor Parallel workers. CPU, QAT, and DSA resources are configured based on this value |
| vLLM | `DP_SIZE` | `1` | Number of Data Parallel groups; total ranks = `TP_SIZE × DP_SIZE` |
| vLLM | `VLLM_CPU_OMP_THREADS_BIND` | Auto-detected from GPU NUMA topology | CPUs for vLLM, one `\|`-separated CPU list per rank |
| IAXL | `KVSHRINK_CPU_OMP_THREADS_BIND` | Same as `VLLM_CPU_OMP_THREADS_BIND` | CPUs for IAXL threads, one entry per rank. Set it for CPU inference; see the [vLLM CPU inference guide](doc/usage/vllm-cpu-inference.en.md) |
| IAXL | `IAXL_RDMA_ENABLE` | `0` | Use a remote KVStore daemon over RDMA (`0`/`1`); see [cross-node KV Cache reuse](#cross-node-kv-cache-reuse) |
| IAXL | `IAXL_RDMA_DAEMON_NIC_IPS` | unset | Daemon RDMA NIC IPs; required when `IAXL_RDMA_ENABLE=1` |
| IAXL | `IAXL_RDMA_CLIENT_NIC_IPS` | unset | Client RDMA NIC IPs; required on the client when `IAXL_RDMA_ENABLE=1` |
| IAXL | `IAXL_OMP_THREAD_NUM` | One per QAT device, one for IAA, plus CPU zip workers | Threads driving compression. Fewer threads use fewer CPU cores |

> [!WARNING]
> Do not enable `IAXL_DSA_GD_ENABLE` on GPUs that do not support P2P DMA. Keep it set to `0`.

## KVShrink vLLM Example

KVShrink is a vLLM V1 KV connector based on IAXL `KVStore`. Configure `setvars.sh`, then start the container:

```bash
./start.sh
```

`setvars.sh` automatically configures the CPU, QAT, and DSA resources for each rank based on the NUMA topology of the first `TP_SIZE` GPUs.

Inside the container, optionally install the package with pip:

```bash
pip install -e . --verbose --no-build-isolation
```

Start the service inside the container:

```bash
./examples/kvshrink-vllm-serve.sh
```

This script starts vLLM on `localhost:8000`, loads `KVShrinkConnector`, and writes logs to `log.kvshrink-vllm`. Use the `MODEL`, `TP_SIZE`, and per-rank CPU/QAT/DSA settings in the startup log to verify the active topology.

Open the same container from a second host terminal:

```bash
docker exec -it -w "$PWD" iaxl.vllm bash
```

Send a Chat Completions test request:

```bash
./tests/vllm-test.sh
```

## CPU Inference With QAT

Build with `DEVICE=cpu` to run the model on CPU while offloading KV cache
compression and decompression to QAT (or IAA). This uses the same `KVStore`,
asynchronous native queues, codecs, and persist/evict APIs as GPU inference.
With the vLLM CPU layout (blocks on axis 0) the codec reads and writes KV blocks
in place, so no scratch buffers are used; byte shuffle or lossy truncation, which
rewrite the codec input, fall back to scratch buffers. CUDA, SYCL and GDRCopy are
not required.

Use a CPU-enabled PyTorch/vLLM environment. Install the Python build requirements
and native development dependencies, including a C/C++ compiler, CMake, NASM,
SQLite, and zlib. NASM is needed by the existing QPL dependency even when IAA is
disabled at runtime.

For the in-tree driver with system qatlib (see Host Setup), build with
`IAXL_QATLIB=intree`:

```bash
python -m pip install -r requirements.txt
DEVICE=cpu IAXL_QATLIB=intree python -m pip install -e . --no-build-isolation
```

For the out-of-tree driver package, leave `IAXL_QATLIB` unset (`oot`). That build
additionally needs Boost/Boost Regex, SSL, udev, and netlink development packages.
Match the user-space library to the host driver: the out-of-tree runtime expects
`/dev/qat_dev_processes`; qatlib uses the host's VFIO setup. Do not install the
GPU-only GDRCopy/DSA components for CPU inference.

Start CPU serving with the standalone CPU launcher, without sourcing the
GPU-topology-based `setvars.sh`:

```bash
MODEL=Qwen/Qwen3-0.6B taskset -c 0-31 ./examples/kvshrink-vllm-cpu-serve.sh
```

The launcher splits the CPUs it may run on evenly between ranks and gives the last
`IAXL_CORES` (default 2) of each rank's share to IAXL; set `VLLM_CPU_OMP_THREADS_BIND`,
`KVSHRINK_CPU_OMP_THREADS_BIND` and `OMP_NUM_THREADS` together to place them
yourself (see the [vLLM CPU inference guide](doc/usage/vllm-cpu-inference.en.md)). It uses
QAT-only compression, BF16, one worker, block size 32, and port
8000; `KVSHRINK_CONNECTOR=0` serves plain vLLM with its own prefix cache instead.
`VLLM_CPU_KVCACHE_SPACE` controls the live CPU KV cache size in GiB;
`IAXL_DDR_POOL_SIZE_GB` independently controls the compressed cache. CPU persisted
data lives under `IAXL_CACHE_DIR/cpu/{compressed,raw}` to avoid reusing incompatible
GPU tensor layouts. Compression format and lossless/lossy settings are unchanged.

### Core budget

On CPU every core the cache path occupies is a core the model does not have, so
the two are partitioned explicitly:

- `VLLM_CPU_OMP_THREADS_BIND` places the inference OpenMP threads.
- `KVSHRINK_CPU_OMP_THREADS_BIND` (one `|`-separated entry per rank, for example
  `30-31|62-63`) pins every IAXL native thread of that rank: the codec pollers, the
  transfer and record queue threads. The connector refuses to start when more than
  one codec thread would share the rank's inference CPUs (unset or overlapping), and
  warns for a single thread. For multiple ranks, also set `TP_SIZE` and
  `KVSHRINK_QAT_DEVICES` (for example, `0|1`).
- `IAXL_USE_OMP=0` (the launcher default) runs IAXL on its own thread pool, off
  vLLM's OpenMP runtime; idle pool threads sleep.
- One codec thread per QAT device (and one for IAA) polls all of that backend's
  instances; `IAXL_OMP_THREAD_NUM` overrides the count, independently of
  inference's `OMP_NUM_THREADS`.

A CPU attention block has `2 * num_kv_heads * block_size * head_size * dtype_bytes`
bytes per layer. Keep it within `IAXL_ZIP_SRC_CAP`, and size `IAXL_ZIP_DST_CAP` for
the compressed output, or reduce `BLOCK_SIZE`. Both capacities default to 256 KiB.

Run the CPU regression suite without QAT hardware, or require QAT-only operation:

```bash
python -u -m unittest discover -s tests -p test_cpu_inference.py -v
IAXL_TEST_ZIP_BACKEND=qat \
	python -u -m unittest discover -s tests -p test_cpu_inference.py -v
```

The tests cover block layouts, FP32/FP16/BF16, raw and mixed-layer compression,
asynchronous waits, native-thread affinity, and persisted reloads. An opt-in inference
smoke test uses a cached `Qwen/Qwen3-0.6B` snapshot (or `IAXL_TEST_MODEL`, a local
model path or cached model ID), disables vLLM prefix caching, and requires actual
external-cache hits and identical cold/warm generated token IDs:

```bash
IAXL_TEST_ZIP_BACKEND=qat IAXL_TEST_VLLM=1 \
	python -u -m unittest discover -s tests -p test_cpu_inference.py -v
```

The smoke test uses async loading by default; set `IAXL_TEST_ASYNC_LOAD_LAYERS=0`
to check synchronous loading. This is a correctness check, not a performance
benchmark. Check for other workloads before running hardware tests.

## KVShrink vLLM Benchmark

Keep the KVShrink vLLM service running and execute the online serving benchmark in the second container terminal:

```bash
./tests/vllm-benchmark.sh
```

## REST API

The management API listens on `localhost:18700` by default and forwards requests to each rank.

| Endpoint | Description |
| --- | --- |
| `GET /v1/cache/status` | Query cache status |
| `POST /v1/cache/evict` | Evict cache groups from DDR |
| `POST /v1/cache/persist` | Persist cache groups to disk |

```bash
curl http://localhost:18700/v1/cache/status
```

For `persist` and `evict`, `count` specifies the maximum number of cache groups to process. To preserve cached data, call `persist` before `evict`:

```bash
curl -X POST http://localhost:18700/v1/cache/persist \
	-H 'Content-Type: application/json' \
	-d '{"count":999999}'
curl -X POST http://localhost:18700/v1/cache/evict \
	-H 'Content-Type: application/json' \
	-d '{"count":999999}'
```

## Cross-Node KV Cache Reuse

remote_pool supports storing KV Cache on a separate daemon node and transferring or reusing it with vLLM client nodes over RDMA. See the [cross-node KV Cache guide](doc/usage/remote-pool-kvstore.en.md) for the topology, build and startup steps, per-rank NIC configuration, and operations.
