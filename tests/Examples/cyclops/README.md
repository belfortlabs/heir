# Cyclops CPU client

Cyclops emission produces four files with
`--cheddar-emitc-entry-interface=runtime=cyclops`: `client_header`,
`client_source`, `server_header`, and `server_source` are the
`heir-translate --mlir-to-cpp --file-id=...` selectors. Their names are
`<entry>_client.h/.cpp` and `<entry>_server.h/.cpp`, with public APIs in
`heir::generated::<entry>::client` and `::server`.

The client links Cyclops' `cyclops_client` CMake target, and the server links
`cyclops`. They belong in separate executables: both runtimes define the same
Cyclops symbols. Only the server requires CUDA. Include HEIR's `include/`
directory (or the wheel's `heir/include`), where the runtime headers live as
`heir/runtime/*.h`, and use Cyclops' CMake targets to obtain its serialization
dependencies and generated headers.

The client calls `Setup` and `KeyGen`, generating the evaluation keys requested
by the compiler's planner. The server calls `Setup` and `Preprocess`. The
helpers in `heir::cyclops` serialize evaluation keys (`writeKeys` / `readKeys`)
and input/output aggregates (`writeValues` / `readValues<Aggregate>`). These
helpers take `context.param_` and a stream and use Cyclops serialization without
a HEIR envelope. Keys are serialized one at a time. The client retains its
secret key.

`Evaluate` takes a `DebugSink` pointer (pass `nullptr` for none). Its checkpoint
contains borrowed ciphertext pointers valid during the callback. The caller
handles serialization, transport, and client-side decryption; the GPU evaluator
needs no secret key.

This standalone coverage test builds the generated client and checks ciphertext
serialization and decryption without evaluating the model. It needs no CUDA
toolkit or GPU. Use the Cyclops revision pinned in `bazel/cyclops/version.bzl`.

```sh
CC=clang CXX=clang++ cmake -S tests/Examples/cyclops -B /tmp/heir-cyclops-client \
  -DCYCLOPS_SOURCE_DIR=/path/to/pinned/cyclops \
  -DHEIR_BIN_DIR=/path/to/heir/bazel-bin/tools \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/heir-cyclops-client --target client_roundtrip -j8
ctest --test-dir /tmp/heir-cyclops-client --output-on-failure
```

Clang needs a C++20 standard library; Cyclops also requires libtommath. CMake
fetches Cyclops' pinned Cereal, JSON, and Protobuf dependencies. Medusa supplies
separate persistent client/server runners and chunked local-process transport.

The maintained AWS task `sky/cyclops_client.yaml` runs this build with HEIR's
hermetic Clang on a CUDA-free host. Supply a `git archive` of the selected
Cyclops revision as `/tmp/cyclops-source.tar.gz`.

Medusa's `test/test_cyclops_integration.py` exercises generated key/ciphertext
exchange, preprocessing, and debug checkpoints. On a configured GPU host, run
`MEDUSA_CYCLOPS_INTEGRATION=1 uv run python -m unittest discover -s test -p test_cyclops_integration.py`.
It uses installed HEIR tools unless `MEDUSA_HEIR_BIN_DIR` is explicitly set.

<!-- mdformat global-off -->
