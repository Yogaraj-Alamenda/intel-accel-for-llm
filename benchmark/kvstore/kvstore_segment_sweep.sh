#!/usr/bin/env bash
#
# Sweeps IAXL_ZIP_SEGMENT_SIZE_KB for QAT and IAA. For every segment size from 16 KiB to
# 1 MiB, plus whole blocks (0), it runs the KVStore benchmark with one backend at a time and
# prints the compression ratio, compress and decompress throughput. Run inside the dev container.
#
# Usage: benchmark/kvstore/kvstore_segment_sweep.sh [options] [-- kvstore_benchmark.py options]
#   --backends LIST   qat, iaa or qat,iaa (default qat,iaa)
#   --sizes LIST      segment sizes in KiB (default 16,32,64,128,256,512,1024)
#   --block-kb N      KV block size in KiB, a multiple of 4 (default 2048)
#   --layers N        KV layers per run (default 4)
#   --reps N          runs per size, averaged (default 1)
#   --data-source S   mock, model or file (default mock)
# The environment is passed on, e.g. IAXL_QAT_ZIP_INSTANCES_PER_DEVICE or IAXL_OMP_THREAD_NUM.

set -euo pipefail

TOP_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
BENCH="$TOP_DIR/benchmark/kvstore/kvstore_benchmark.sh"

BACKENDS=qat,iaa
SIZES=16,32,64,128,256,512,1024
BLOCK_KB=2048
LAYERS=4
REPS=1
DATA_SOURCE=mock
EXTRA=()

usage() {
    sed -n 's/^# \{0,1\}//; 7,14p' "${BASH_SOURCE[0]}"
    exit "${1:-0}"
}

while (($#)); do
    case "$1" in
        --backends) BACKENDS=$2; shift 2 ;;
        --sizes) SIZES=$2; shift 2 ;;
        --block-kb) BLOCK_KB=$2; shift 2 ;;
        --layers) LAYERS=$2; shift 2 ;;
        --reps) REPS=$2; shift 2 ;;
        --data-source) DATA_SOURCE=$2; shift 2 ;;
        -h | --help) usage ;;
        --) shift; EXTRA=("$@"); break ;;
        *) echo "ERROR: unknown option $1" >&2; usage 2 ;;
    esac
done

if ! python3 -c 'import torch' 2>/dev/null; then
    echo "ERROR: torch not found; run inside the dev container, e.g." >&2
    echo "  docker exec -w \"\$PWD\" iaxl.vllm bash benchmark/kvstore/kvstore_segment_sweep.sh" >&2
    exit 1
fi
if ! [[ "$BLOCK_KB" =~ ^[0-9]+$ ]] || ((BLOCK_KB < 4 || BLOCK_KB % 4)); then
    echo "ERROR: --block-kb must be a positive multiple of 4" >&2
    exit 2
fi

# One block is K and V of TOKENS tokens x 8 heads x 128 dims in bf16, i.e. TOKENS x 4 KiB.
TOKENS=$((BLOCK_KB / 4))
# Even blocks are tested, so each layer moves CACHE_BLOCKS / 2 blocks: about 128 MiB.
CACHE_BLOCKS=$((2 * 128 * 1024 / BLOCK_KB))
((CACHE_BLOCKS < 2)) && CACHE_BLOCKS=2
# A whole block must fit in one request.
CAP=$((BLOCK_KB * 1024 > 4194304 ? BLOCK_KB * 1024 : 4194304))
export IAXL_ZIP_SRC_CAP=${IAXL_ZIP_SRC_CAP:-$CAP} IAXL_ZIP_DST_CAP=${IAXL_ZIP_DST_CAP:-$CAP}

ROWS=()
for s in ${SIZES//,/ }; do
    if ! [[ "$s" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: segment size '$s' is not a positive integer" >&2
        exit 2
    fi
    if ((s >= BLOCK_KB)); then
        echo "NOTE: skipping ${s} KiB, not smaller than the ${BLOCK_KB} KiB block" >&2
        continue
    fi
    ROWS+=("$s")
done
ROWS+=(0)

RESULTS=$(mktemp)
CONFIG=$(mktemp)
trap 'rm -f "$RESULTS" "$CONFIG"' EXIT

# Prints "ratio zip unzip" of one benchmark run, or nothing when it fails.
run_one() {
    local backend=$1 seg=$2 out
    out=$(IAXL_ZIP_SEGMENT_SIZE_KB=$seg bash "$BENCH" "--$backend" \
        --shape 2 "$CACHE_BLOCKS" "$TOKENS" 8 128 --num-layers "$LAYERS" \
        --data-source "$DATA_SOURCE" "${EXTRA[@]}" 2>&1) || return 0
    grep -m1 -o 'qat_zip=.*omp_threads=[0-9]*' <<<"$out" >"$CONFIG" || true
    grep -q '^Verification: passed' <<<"$out" || return 0
    awk '/^Compression ratio:/ {r = $3 + 0}
         /^Compress throughput:/ {z = $3}
         /^Decompress throughput:/ {u = $3}
         END {if (r && z && u) print r, z, u}' <<<"$out"
}

echo "Block ${BLOCK_KB} KiB, ${LAYERS} layers, data ${DATA_SOURCE}, ${REPS} run(s) per size"
for backend in ${BACKENDS//,/ }; do
    for seg in "${ROWS[@]}"; do
        printf '  %s segment %s KiB ...\r' "$backend" "$seg" >&2
        for ((r = 0; r < REPS; r++)); do
            res=$(run_one "$backend" "$seg")
            echo "$backend $seg ${res:-fail}" >>"$RESULTS"
        done
    done
    printf '%40s\r' "" >&2
    if [[ -s $CONFIG ]]; then echo "$backend: $(cat "$CONFIG")"; fi
done
echo

awk -v block="$BLOCK_KB" -v backends="$BACKENDS" '
{
    key = $1 SUBSEP $2
    if (!(($2) in seen)) { seen[$2] = 1; rows[++nrows] = $2 }
    if ($3 == "fail") { fails[key]++; next }
    n[key]++; ratio[key] += $3; zip[key] += $4; unzip[key] += $5
}
END {
    nb = split(backends, bl, ",")
    for (b = 1; b <= nb; b++) {
        be = bl[b]
        printf "%s\n", toupper(be)
        printf "| Segment KiB | Ratio  | Zip GB/s | Unzip GB/s |\n"
        printf "|-------------|--------|----------|------------|\n"
        for (i = 1; i <= nrows; i++) {
            s = rows[i]; key = be SUBSEP s
            label = s == 0 ? sprintf("0 (%d)", block) : s
            if (!n[key]) { printf "| %-11s | failed |          |            |\n", label; continue }
            r = ratio[key] / n[key]; z = zip[key] / n[key]; u = unzip[key] / n[key]
            printf "| %-11s | %.3fx | %8.2f | %10.2f |\n", label, r, z, u
        }
        print ""
    }
}' "$RESULTS"
