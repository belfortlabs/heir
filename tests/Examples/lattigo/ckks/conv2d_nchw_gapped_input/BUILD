load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "conv2d_nchw_gapped_input",
    go_library_name = "conv2dnchwgappedinput",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=min-slot-count=4096 scaling-mod-bits=45 first-mod-bits=60",
        "--scheme-to-lattigo",
    ],
    mlir_src = "conv2d_nchw_gapped_input.mlir",
)

go_test(
    name = "conv2d_nchw_gapped_input_test",
    srcs = ["conv2d_nchw_gapped_input_test.go"],
    embed = [":conv2dnchwgappedinput"],
    deps = [":conv2dnchwgappedinput_utils"],
)
