load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "depthwise_conv2d",
    go_library_name = "depthwiseconv2d",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=min-slot-count=4096 scaling-mod-bits=45 first-mod-bits=60",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:depthwise_conv2d.mlir",
)

go_test(
    name = "depthwiseconv2d_test",
    size = "small",
    srcs = ["depthwise_conv2d_test.go"],
    embed = [":depthwiseconv2d"],
    deps = [":depthwiseconv2d_utils"],
)
