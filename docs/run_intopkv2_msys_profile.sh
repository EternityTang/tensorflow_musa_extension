#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

MSYS_OUTPUT="${MSYS_OUTPUT:-${REPO_ROOT}/test/ops/benchmark_results/intopkv2_${TIMESTAMP}}"
DEVICE="${DEVICE:-0}"

mkdir -p "$(dirname "${MSYS_OUTPUT}")"

BENCHMARK_CMD=(
  python "${REPO_ROOT}/test/ops/intopkv2_benchmark.py"
  "$@"
)

printf 'MSYS_OUTPUT=%s\n' "${MSYS_OUTPUT}"
printf 'Running benchmark command:'
printf ' %q' "${BENCHMARK_CMD[@]}"
printf '\n'

msys profile \
  --device="${DEVICE}" \
  --trace=musa,osrt \
  --output="${MSYS_OUTPUT}" \
  "${BENCHMARK_CMD[@]}"

msys stats \
  --format csv \
  --report musa_gpu_kern_sum,musa_kern_exec_sum,musa_api_gpu_sum \
  --output "${MSYS_OUTPUT}" \
  "${MSYS_OUTPUT}.msys-rep"

printf 'Profile report: %s.msys-rep\n' "${MSYS_OUTPUT}"
printf 'Stats CSV prefix: %s\n' "${MSYS_OUTPUT}"
