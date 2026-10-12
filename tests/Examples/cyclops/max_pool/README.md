# Cyclops max pool end-to-end test

This test compiles `max_pool.mlir` with HEIR and runs it on Cyclops. The model
is a ceil-mode `MaxPool1d(8)` over 4 channels of 61 values in `[-3, 5]`, in
the form that torch-mlir produces: a `-inf` pad, then
`linalg.pooling_ncw_max`. It checks these steps together:

- `--linalg-canonicalizations` replaces the `-inf` pad with the domain's lower
  bound.
- HEIR passes `value_bound = 5` to Cyclops' `MaxPool`, and does not normalize
  the input.
- The level plan: the generated code asserts the input and output levels of the
  pool at run time.
- The key plan of the pool, from Cyclops' CPU planner.

The client decrypts the result and compares it with a cleartext max pool. The
test accepts an error of 0.1, which is Cyclops' own MaxPool tolerance (0.01 in
its comparison domain) times `2 * value_bound`. The last window of each channel
holds only negative values, so a wrong padding value gives a wrong maximum.

## Requirements

- A CUDA GPU for the server. The client needs no GPU.
- A Cyclops checkout of branch `aikata/maxpool` at 0e5dde7b or later. HEIR
  must be built against the same revision. `bazel/cyclops/version.bzl` pins
  0e5dde7b. For another checkout, use
  `--override_repository=+cyclops_deps+cyclops=<checkout>` when the checkout
  has the `REPO.bazel` and `BUILD.bazel` of `bazel/cyclops/cyclops.BUILD`.
- CMake 3.30 or later, Ninja, Clang with a C++20 standard library, and
  libtommath.

A window of fewer than 8 slots, or a ciphertext of fewer than 256 slots, needs
the Cyclops fixes for requirements R2 and R6 in
`cyclops_maxpool_requirements.md`. Without them, key planning stops with an
error for these pools.

## Run

```sh
bazel build //tools:heir-opt //tools:heir-translate
tests/Examples/cyclops/max_pool/run_max_pool_e2e.sh \
  /path/to/cyclops bazel-bin/tools /path/with/space/heir-max-pool
```

The script configures and builds the client and the server separately, because
the CPU and GPU runtimes define the same Cyclops symbols. It then starts the
client and the server, which exchange the keys, the encrypted input and the
encrypted result as files in `BUILD_DIR/exchange`. The evaluation keys take
about 4.3 GB on disk, so select a `BUILD_DIR` with space. The exit status is the
result of the client's comparison. `MAX_POOL_E2E_TIMEOUT_S` sets how long each
side waits for a file (default 3600 seconds).

## Known issues

- The server builds for `CUDA_ARCHITECTURES` (default `native`). With
  several GPUs in the host, `native` also targets the others, and the link
  fails when a Cyclops dependency (cuPQC) does not support one of them, for
  example sm_75. Then set `CUDA_ARCHITECTURES` to the architecture of the GPU
  that runs the server, for example `120`.
- CUDA numbers the GPUs fastest first by default. To select a GPU by its
  `nvidia-smi` index, set `CUDA_DEVICE_ORDER=PCI_BUS_ID` together with
  `CUDA_VISIBLE_DEVICES`. On a GPU with 4 GB, the server runs out of memory.
- The script uses `CC` and `CXX` (default `clang` and `clang++`) also as the
  CUDA host compiler. CUDA 13.2 rejects Clang 22 and later. Then set, for
  example, `CC=gcc-15 CXX=g++-15`.
- The server prints "WARN: Rescale is not adequate for in-place operations"
  for each in-place `Rescale` in the generated code. Cyclops rescales into a
  temporary in that case, so the result is correct.

On 2026-10-05, this test passed on an RTX 5090 with a worst error of 0.00035.
The client and the server each took about 80 seconds, and the client used
17 GB of memory.

On 2026-10-09, it passed on an RTX 5090 with a worst error of 0.00022, against
Cyclops 0e5dde7b with the patch for R2 and R6, built with gcc-15.
