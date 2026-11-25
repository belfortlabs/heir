load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "LinearOp",
    srcs = ["LinearOp.cpp"],
    hdrs = ["LinearOp.h"],
    deps = [
        "@heir//lib/Utils/Layout:LinearSpec",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "StridedLayoutPropagation",
    srcs = ["StridedLayoutPropagation.cpp"],
    hdrs = ["StridedLayoutPropagation.h"],
    deps = [
        ":LinearOp",
        ":pass_inc_gen",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Utils:AttributeUtils",
        "@heir//lib/Utils/Layout:LinearSpec",
        "@heir//lib/Utils/Layout:StridedLayout",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "DecomposeMaxPool",
    srcs = ["DecomposeMaxPool.cpp"],
    hdrs = ["DecomposeMaxPool.h"],
    deps = [
        ":decompose_max_pool_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "CountOps",
    srcs = ["CountOps.cpp"],
    hdrs = ["CountOps.h"],
    deps = [
        ":count_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
    ],
)

add_heir_transforms(
    generated_target_name = "count_ops_inc_gen",
    pass_name = "CountOps",
    td_file = "CountOps.td",
)

add_heir_transforms(
    generated_target_name = "decompose_max_pool_inc_gen",
    pass_name = "DecomposeMaxPool",
    td_file = "DecomposeMaxPool.td",
)

add_heir_transforms(
    pass_name = "StridedLayoutPropagation",
    td_file = "StridedLayoutPropagation.td",
)
