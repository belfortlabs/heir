load("@heir//lib/Target/CompilationTarget:defs.bzl", "heir_backend_config")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

heir_backend_config(
    name = "cheddar_backend_config",
    srcs = ["TargetConfig.td"],
)

# The Cyclops MaxPoolConfig of a kernel.max_pool. Include it only from targets
# that define CYCLOPS_PLANNER_BUILD, as the planner library does.
cc_library(
    name = "MaxPoolCyclopsConfig",
    hdrs = ["MaxPoolCyclopsConfig.h"],
    tags = ["no-remote"],
    deps = [
        "@cyclops//:planner",
        "@heir//lib/Utils:MaxPoolUtils",
    ],
)

# Calls the CPU-only Cyclops planner library, which reports errors with
# exceptions and is built on the trusted host.
cc_library(
    name = "MaxPoolPlan",
    srcs = ["MaxPoolPlan.cpp"],
    hdrs = ["MaxPoolPlan.h"],
    copts = ["-fexceptions"],
    implementation_deps = [
        ":MaxPoolCyclopsConfig",
        "@cyclops//:planner",
    ],
    linkopts = select({
        "@platforms//os:macos": ["-Wl,-rpath,@loader_path"],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    local_defines = ["CYCLOPS_PLANNER_BUILD"],
    tags = ["no-remote"],
    deps = ["@heir//lib/Utils:MaxPoolUtils"],
)

cc_test(
    name = "MaxPoolPlanTest",
    srcs = ["MaxPoolPlanTest.cpp"],
    tags = ["no-remote"],
    deps = [
        ":MaxPoolPlan",
        "@googletest//:gtest_main",
    ],
)

# The level model of kernel.max_pool, attached as external models so that the
# Kernel dialect does not depend on the Cyclops planner.
cc_library(
    name = "MaxPoolInterfaces",
    srcs = ["MaxPoolInterfaces.cpp"],
    hdrs = ["MaxPoolInterfaces.h"],
    deps = [
        ":MaxPoolPlan",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@heir//lib/Dialect/Kernel/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Utils:MaxPoolUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Support",
    ],
)
