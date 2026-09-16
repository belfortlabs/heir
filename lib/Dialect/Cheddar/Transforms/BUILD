load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "CheddarBufferize",
    srcs = ["CheddarBufferize.cpp"],
    hdrs = ["CheddarBufferize.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "BufferizableOpInterfaceImpl",
    srcs = ["BufferizableOpInterfaceImpl.cpp"],
    hdrs = ["BufferizableOpInterfaceImpl.h"],
    deps = [
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@heir//lib/Dialect/Cheddar/IR:ops_inc_gen",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationInterfaces",
        "@llvm-project//mlir:DestinationStyleOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "BuildEntryInterface",
    srcs = ["BuildEntryInterface.cpp"],
    hdrs = ["BuildEntryInterface.h"],
    deps = [
        ":build_entry_interface_inc_gen",
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@heir//lib/Dialect/Preprocessing/IR:Dialect",
        "@heir//lib/Utils:EntryInterfaceUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "ConfigureCryptoContext",
    srcs = ["ConfigureCryptoContext.cpp"],
    hdrs = ["ConfigureCryptoContext.h"],
    deps = [
        ":configure_crypto_context_inc_gen",
        "@heir//lib/Analysis/RotationAnalysis",
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@heir//lib/Utils:TransformUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "PlanEvaluationKeys",
    srcs = ["PlanEvaluationKeys.cpp"],
    hdrs = ["PlanEvaluationKeys.h"],
    copts = ["-fexceptions"],
    implementation_deps = ["@cyclops//:planner"],
    linkopts = select({
        "@platforms//os:macos": ["-Wl,-rpath,@loader_path"],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    local_defines = ["CYCLOPS_PLANNER_BUILD"],
    tags = ["no-remote"],
    deps = [
        ":plan_evaluation_keys_inc_gen",
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "FuseOps",
    srcs = ["FuseOps.cpp"],
    hdrs = ["FuseOps.h"],
    deps = [
        ":fuse_ops_inc_gen",
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

add_heir_transforms(
    generated_target_name = "build_entry_interface_inc_gen",
    header_filename = "BuildEntryInterface.h.inc",
    pass_name = "CheddarBuildEntryInterface",
    td_file = "BuildEntryInterface.td",
)

add_heir_transforms(
    generated_target_name = "configure_crypto_context_inc_gen",
    header_filename = "ConfigureCryptoContext.h.inc",
    pass_name = "CheddarConfigureCryptoContext",
    td_file = "ConfigureCryptoContext.td",
)

add_heir_transforms(
    generated_target_name = "plan_evaluation_keys_inc_gen",
    header_filename = "PlanEvaluationKeys.h.inc",
    pass_name = "CheddarPlanEvaluationKeys",
    td_file = "PlanEvaluationKeys.td",
)

add_heir_transforms(
    generated_target_name = "fuse_ops_inc_gen",
    header_filename = "FuseOps.h.inc",
    pass_name = "CheddarFuseOps",
    td_file = "FuseOps.td",
)
