load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "conv_conv_transpose2d",
    go_library_name = "convconvtranspose2d",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=min-slot-count=4096 scaling-mod-bits=45 first-mod-bits=60 greedy-level-budget=20",
        "--scheme-to-lattigo",
    ],
    mlir_src = "conv_conv_transpose2d.mlir",
)

go_test(
    name = "conv_conv_transpose2d_test",
    srcs = ["conv_conv_transpose2d_test.go"],
    embed = [":convconvtranspose2d"],
)

heir_lattigo_lib(
    name = "conv_transpose2d_k4x3_s2x1",
    go_library_name = "convtranspose2dk4x3s2x1",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=min-slot-count=4096 scaling-mod-bits=45 first-mod-bits=60",
        "--scheme-to-lattigo",
    ],
    mlir_src = "conv_transpose2d_k4x3_s2x1.mlir",
)

go_test(
    name = "conv_transpose2d_k4x3_s2x1_test",
    srcs = ["conv_transpose2d_k4x3_s2x1_test.go"],
    embed = [":convtranspose2dk4x3s2x1"],
)
