Setuptools handles package metadata, wheel files, and source archives. The native `build_ext` command reuses a verified binary payload before it invokes Bazel or CMake.

The helper accepts a wheel only when its commit, checksum, platform, and build profile match. The helper copies only the configured native files. The current source supplies all Python files and package metadata.

Set `BELFORT_WHEEL_DIR` to a local directory of wheels and manifests. Otherwise, the helper checks the GitHub release `wheels-<full-commit>`. Set `TOOLCHAIN_GITHUB_TOKEN` for private release access.

Each wheel contains `<module>/toolchain.json`. Run `uv run --no-project --with setuptools --with packaging python build_support/source_wheels.py dist` to create the manifest. Run this command after each build in a shared artifact directory.

Set `BELFORT_FORCE_SOURCE=1` for a release build or a custom native configuration. Set `BELFORT_REQUIRE_WHEEL=1` to prohibit native compilation. Custom build settings require `BELFORT_FORCE_SOURCE=1` because setuptools owns the PEP 517 settings.

An existing uv cache entry can bypass the native build command. Use separate uv caches for different native profiles. Reinstall the package when a profile changes. A force-source check also requires a fresh uv cache and package reinstallation.

The source archive contains `build-source.json` to retain the source identity without Git. Dirty checkouts bypass wheel reuse. Publication rejects wheels from dirty checkouts.

HEIR accepts `HEIR_BAZELRC` as an absolute path to a private Bazel configuration file. This file applies to fetch, build, and shutdown. HEIR Git builds use `0.0.0+g<full-commit>` as the package version. Explicit setuptools-scm overrides control release versions. Setuptools-scm retains the published version from an unpacked release archive.

Cyclops accepts `CYCLOPS_CUDA_VERSION`, `CYCLOPS_CUDA_ARCHITECTURES`, and `CYCLOPS_BUILD_JOBS`. Its default profile uses CUDA 13.0 with architectures 86 and 89. The wheel requires Linux x86-64, the declared CUDA runtime, libtommath, libquadmath, libgomp, and the system C++ runtime. It does not include a GPU driver.
