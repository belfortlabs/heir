load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ParameterFile",
    srcs = ["ParameterFile.cpp"],
    hdrs = ["ParameterFile.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "ParameterFileTest",
    srcs = ["ParameterFileTest.cpp"],
    deps = [
        ":ParameterFile",
        "@googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)
