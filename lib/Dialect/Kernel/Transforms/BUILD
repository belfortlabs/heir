# Kernel dialect transforms

load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = ["Passes.h"],
    deps = [
        ":ImplementLinearTransforms",
        ":PrepareLinearTransforms",
        ":pass_inc_gen",
        "@heir//lib/Dialect/Kernel/IR:Dialect",
    ],
)

cc_library(
    name = "PrepareLinearTransforms",
    srcs = ["PrepareLinearTransforms.cpp"],
    hdrs = ["PrepareLinearTransforms.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/Kernel/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Target/CompilationTarget",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "ImplementLinearTransforms",
    srcs = ["ImplementLinearTransforms.cpp"],
    hdrs = ["ImplementLinearTransforms.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Kernel/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Kernel:AbstractValue",
        "@heir//lib/Kernel:ArithmeticDag",
        "@heir//lib/Kernel:IRMaterializingVisitor",
        "@heir//lib/Kernel:LinearTransformImplementation",
        "@heir//lib/Kernel:Utils",
        "@heir//lib/Target/CompilationTarget",
        "@heir//lib/Utils/Layout:LinearSpec",
        "@heir//lib/Utils/Layout:SparseBsgs",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

add_heir_transforms(
    header_filename = "Passes.h.inc",
    pass_name = "Kernel",
    td_file = "Passes.td",
)
