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

# Loads Cyclops' proprietary planner library at run time. heir-opt does not
# link it; it finds libcyclops_planner on the library path.
cc_library(
    name = "CyclopsPlanner",
    srcs = ["CyclopsPlanner.cpp"],
    hdrs = ["CyclopsPlanner.h"],
    linkopts = select({
        "@platforms//os:macos": ["-Wl,-rpath,@loader_path"],
        "//conditions:default": [
            "-ldl",
            "-Wl,-rpath,$$ORIGIN",
        ],
    }),
    deps = ["@heir//bazel/cyclops:planner"],
)

# The Cyclops MaxPoolConfig of a kernel.max_pool.
cc_library(
    name = "MaxPoolCyclopsConfig",
    hdrs = ["MaxPoolCyclopsConfig.h"],
    deps = [
        ":CyclopsPlanner",
        "@heir//lib/Utils:MaxPoolUtils",
    ],
)

# The level requirements of a max pool, from Cyclops' planner.
cc_library(
    name = "MaxPoolPlan",
    srcs = ["MaxPoolPlan.cpp"],
    hdrs = ["MaxPoolPlan.h"],
    implementation_deps = [
        ":CyclopsPlanner",
        ":MaxPoolCyclopsConfig",
    ],
    deps = ["@heir//lib/Utils:MaxPoolUtils"],
)

cc_test(
    name = "MaxPoolPlanTest",
    srcs = ["MaxPoolPlanTest.cpp"],
    # The planner library is on the local machine.
    tags = ["no-remote"],
    deps = [
        ":CyclopsPlanner",
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
        ":CyclopsPlanner",
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
