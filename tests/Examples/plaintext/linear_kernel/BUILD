load("@heir//tests/llvm_runner:llvm_runner.bzl", "llvm_runner_test")

package(default_applicable_licenses = ["@heir//:license"])

llvm_runner_test(
    name = "linear_ops_test",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-size=1024",
    ],
    main_c_src = "linear_ops_test.cpp",
    mlir_src = "linear_ops.mlir",
)

llvm_runner_test(
    name = "linear_ops_unrolled_test",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-size=1024 unroll-fhe-kernel-loops=true",
    ],
    main_c_src = "linear_ops_test.cpp",
    mlir_src = "linear_ops.mlir",
)

llvm_runner_test(
    name = "network_test",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-size=1024",
    ],
    main_c_src = "network_test.cpp",
    mlir_src = "network.mlir",
)

llvm_runner_test(
    name = "layout_ops_test",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-size=1024",
    ],
    main_c_src = "layout_ops_test.cpp",
    mlir_src = "layout_ops.mlir",
)

llvm_runner_test(
    name = "max_pool_test",
    heir_opt_flags = [
        "--decompose-max-pool",
        "--heir-polynomial-to-llvm",
    ],
    main_c_src = "max_pool_test.cpp",
    mlir_src = "max_pool.mlir",
)
