#!/usr/bin/env bash
# Builds and runs the max pool end-to-end test (see README.md): the CPU client
# and the GPU server exchange keys, inputs and outputs as files.
#
# Usage: run_max_pool_e2e.sh CYCLOPS_SOURCE_DIR HEIR_BIN_DIR BUILD_DIR
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 CYCLOPS_SOURCE_DIR HEIR_BIN_DIR BUILD_DIR" >&2
  exit 2
fi
cyclops=$(realpath "$1")
heir_bin=$(realpath "$2")
build=$(realpath -m "$3")
source_dir=$(dirname "$(realpath "$0")")
cc=${CC:-clang}
cxx=${CXX:-clang++}

configure_and_build() {
  local side=$1
  shift
  cmake -S "$source_dir" -B "$build/$side" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$cc" -DCMAKE_CXX_COMPILER="$cxx" \
    -DCYCLOPS_SOURCE_DIR="$cyclops" -DHEIR_BIN_DIR="$heir_bin" "$@"
  cmake --build "$build/$side" --target "max_pool_$side"
}

configure_and_build client -DMAX_POOL_SERVER=OFF
configure_and_build server -DMAX_POOL_SERVER=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_CUDA_HOST_COMPILER="$cxx"

# The evaluation keys are several gigabytes, so the exchange directory is in
# BUILD_DIR, not in /tmp.
exchange="$build/exchange"
rm -rf "$exchange"
mkdir -p "$exchange"
trap 'rm -rf "$exchange"' EXIT
# The client needs no GPU.
CUDA_VISIBLE_DEVICES="" "$build/client/max_pool_client" "$exchange" &
client=$!
if ! "$build/server/max_pool_server" "$exchange"; then
  kill "$client" 2>/dev/null || true
  wait "$client" || true
  exit 1
fi
wait "$client"
