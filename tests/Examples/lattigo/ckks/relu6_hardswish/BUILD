load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# End-to-end coverage for ReLU6 and hard-swish on a secret input: the clamps
# are split into two composite ReLUs each, and hard-swish's division by 6
# becomes a plaintext multiplication.
heir_lattigo_lib(
    name = "relu6_hardswish",
    go_library_name = "relu6hardswish",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=min-slot-count=1024 use-composite-relu=true greedy-level-budget=30",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:relu6_hardswish.mlir",
    split_preprocessing = False,
)

go_test(
    name = "relu6hardswish_test",
    srcs = ["relu6_hardswish_test.go"],
    embed = [":relu6hardswish"],
)
