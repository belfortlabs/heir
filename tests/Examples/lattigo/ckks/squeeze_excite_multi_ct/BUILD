load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "squeeze_excite_multi_ct",
    go_library_name = "squeezeexcitemultict",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=min-slot-count=1024",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:squeeze_excite_multi_ct.mlir",
)

go_test(
    name = "squeezeexcitemultict_test",
    size = "small",
    srcs = ["squeeze_excite_multi_ct_test.go"],
    embed = [":squeezeexcitemultict"],
)
