load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

exports_files(["cyclops.BUILD"])

# --//bazel/cyclops:real_planner builds the Cyclops key planner from the private
# Cyclops source at version.bzl's commit. By default heir-opt links a stub with
# the same C API whose planning calls fail; a real libcyclops_planner on the
# library path replaces it at run time.
bool_flag(
    name = "real_planner",
    build_setting_default = False,
)

config_setting(
    name = "use_real_planner",
    flag_values = {":real_planner": "true"},
)

alias(
    name = "planner",
    actual = select({
        ":use_real_planner": "@cyclops//:planner",
        "//conditions:default": ":planner_stub",
    }),
)

# The declarations HEIR uses from Cyclops' cyclops_planner.h.
cc_library(
    name = "planner_api",
    hdrs = ["cyclops_planner.h"],
    includes = ["."],
)

# The stub carries the real library's name, so the loader takes either.
cc_binary(
    name = "libcyclops_planner.so",
    srcs = ["planner_stub.cc"],
    linkopts = ["-Wl,-soname,libcyclops_planner.so"],
    linkshared = True,
    target_compatible_with = select({
        "@platforms//os:macos": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    deps = [":planner_api"],
)

cc_binary(
    name = "libcyclops_planner.dylib",
    srcs = ["planner_stub.cc"],
    linkopts = ["-Wl,-install_name,@rpath/libcyclops_planner.dylib"],
    linkshared = True,
    target_compatible_with = ["@platforms//os:macos"],
    deps = [":planner_api"],
)

cc_import(
    name = "planner_stub_library",
    shared_library = select({
        "@platforms//os:macos": ":libcyclops_planner.dylib",
        "//conditions:default": ":libcyclops_planner.so",
    }),
)

cc_library(
    name = "planner_stub",
    deps = [
        ":planner_api",
        ":planner_stub_library",
    ],
)

# "real" or "stub", for lit's cyclops-planner feature.
genrule(
    name = "planner_kind",
    outs = ["planner_kind.txt"],
    cmd = select({
        ":use_real_planner": "echo real > $@",
        "//conditions:default": "echo stub > $@",
    }),
)
