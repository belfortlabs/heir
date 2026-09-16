The source wheel backend keeps a Git dependency at its pinned commit. It reuses a wheel only after commit, checksum, platform, and build profile checks.

Use `BELFORT_WHEEL_DIR` to supply a local directory of wheels and manifests. Otherwise, the backend checks the GitHub release `wheels-<full-commit>`. Set `TOOLCHAIN_GITHUB_TOKEN` for private release access.

Each wheel contains `<module>/toolchain.json`. Run `uv run --no-project --with setuptools --with wheel --with packaging python build_support/wheel_backend.py dist` to create the manifest.

Set `BELFORT_FORCE_SOURCE=1` for a release build or a custom native configuration. Set `BELFORT_REQUIRE_WHEEL=1` on CI consumers to prohibit native compilation. Use a fresh uv cache when you test either option. An existing uv cache entry can bypass the build backend.

The metadata hook delegates to setuptools without a native build. Wheel reuse preserves the source metadata and the downloaded native payload. This permits a source SCM version to differ from a release version. The backend regenerates the wheel RECORD file after that change.

The source distribution contains `build-source.json` so a wheel build can retain its source identity without Git. Dirty checkouts bypass wheel reuse. Publication rejects wheels from dirty checkouts.

HEIR accepts `HEIR_BAZELRC` as an absolute path to a private Bazel configuration file. The file applies to both fetch and build commands. Use it for BuildBuddy credentials and executor configuration.

Cyclops accepts `CYCLOPS_CUDA_VERSION`, `CYCLOPS_CUDA_ARCHITECTURES`, and `CYCLOPS_BUILD_JOBS`. Its default profile uses CUDA 13.0 with architectures 86 and 89. The wheel requires Linux x86-64, the declared CUDA runtime, libtommath, libquadmath, libgomp, and the system C++ runtime. It does not include a GPU driver. Its metadata preparation does not require CUDA.

The wheel tag describes the Python and platform compatibility. The manifest also records the CUDA profile. Keep separate uv caches when you change the native build profile.

Source builds use `0.0.0+g<full-commit>` as the package version. A new Git tag does not change the source dependency metadata. An explicit setuptools-scm version override still controls release versions.
