load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "planner_sources",
    srcs = ["CMakeLists.txt"] + glob([
        "client/include/**",
        "client/src/core/**",
        "include/**",
        "planner/**",
        "src/extension/**",
    ]),
)

# Configure only planner/, never the CUDA server or standalone client project.
# The installed header and binary come from the same Cyclops revision.
cmake(
    name = "planner",
    build_data = ["@llvm//tools:clang++"],
    cache_entries = {
        "BUILD_PLANNER_TEST": "OFF",
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_CXX_STANDARD": "20",
    },
    # CMake links try-compiles from scratch directories, outside the execroot.
    env = {"LLVM_CLANGXX": "$$EXT_BUILD_ROOT$$/$(execpath @llvm//tools:clang++)"},
    generate_args = ["-GNinja"],
    lib_source = ":planner_sources",
    out_shared_libs = select({
        "@platforms//os:macos": ["libcyclops_planner.dylib"],
        "//conditions:default": ["libcyclops_planner.so"],
    }),
    # Keep proprietary sources and planner artifacts off shared remote workers
    # and caches. This action runs on the trusted build host.
    tags = ["no-remote"],
    working_directory = "planner",
)
