load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "PendingRescaleAnalysis",
    srcs = ["PendingRescaleAnalysis.cpp"],
    hdrs = ["PendingRescaleAnalysis.h"],
    deps = [
        "@heir//lib/Analysis:Utils",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
