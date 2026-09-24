# See README.md for setup required to run these tests

load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# The same dot product as ckks/dot_product_8f, compiled so the client encrypts
# at the bottom of the modulus chain. The generated encryption helpers build
# their plaintexts at level 0 and the entry function bootstraps each argument
# before using it, so decrypting the result to the same value the plain
# version produces is what shows the entry bootstraps are placed and
# parameterized correctly.
heir_lattigo_lib(
    name = "level_zero_encryption",
    go_library_name = "levelzeroencryption",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=min-slot-count=2048 level-zero-encryption=true",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:dot_product_8f.mlir",
)

# For Google-internal reasons we must separate the go_test rules from the macro
# above.
go_test(
    name = "levelzeroencryption_test",
    size = "large",
    srcs = ["level_zero_encryption_test.go"],
    embed = [":levelzeroencryption"],
)
