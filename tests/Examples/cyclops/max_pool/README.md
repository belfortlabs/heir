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
- A Cyclops checkout with `extension/max/MaxPoolPlanner.h`, for example branch
  `aikata/maxpool` at e8efd441 or later. HEIR must be built against the same
  revision. `bazel/cyclops/version.bzl` pins e8efd441. For another checkout,
  use `--override_repository=+cyclops_deps+cyclops=<checkout>` when the
  checkout has the `REPO.bazel` and `BUILD.bazel` of
  `bazel/cyclops/cyclops.BUILD`.
- CMake 3.30 or later, Ninja, Clang with a C++20 standard library, and
  libtommath.

The windows of 8 slots keep the pool out of Cyclops' small-window league level
cap (requirement R2 in `cyclops_maxpool_requirements.md`). With a smaller
window, the cap lowers the levels when the input level is more than the
selector depth + 1 above the gather stages. Key planning compares
`PlanMaxPool` with HEIR's levels, and stops in that case until Cyclops can
disable the cap.

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

- HEIR emits `BootSecretId()`, but Cyclops e8efd441 has only
  `NativeSecretId()` (requirement R16). Until the emitter changes, build the
  two sides by hand: configure them as the script does, generate
  `max_pool_client.cpp` and `max_pool_server.cpp` with Ninja, replace
  `BootSecretId()` by `NativeSecretId()` in both files, and then build and
  start the two programs.
- CUDA numbers the GPUs fastest first by default. To select a GPU by its
  `nvidia-smi` index, set `CUDA_DEVICE_ORDER=PCI_BUS_ID` together with
  `CUDA_VISIBLE_DEVICES`. On a GPU with 4 GB, the server runs out of memory.
- The server prints "WARN: Rescale is not adequate for in-place operations"
  for each in-place `Rescale` in the generated code. Cyclops rescales into a
  temporary in that case, so the result is correct.

On 2026-10-05, this test passed on an RTX 5090 with a worst error of 0.00035.
The client and the server each took about 80 seconds, and the client used
17 GB of memory.
