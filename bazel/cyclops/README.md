# Cyclops planner dependency

HEIR links the CPU-only Cyclops planner to determine which evaluation keys
generated clients must create. Bazel builds `planner/` through
`rules_foreign_cc`.

The source revision is pinned in `version.bzl`. Fetching it requires SSH access
to `belfortlabs/cyclops`. Update this pin when adopting a new planner revision.

Build the dependency with `bazel build @cyclops//:planner`, or build HEIR
normally. The target exposes the installed header and shared library together.
Bazel supplies the library to binaries and tests; wheel packaging copies it
beside the installed HEIR tools. These artifacts contain the proprietary planner
and are intended for trusted compiler environments such as Perseus.

<!-- mdformat global-off -->
