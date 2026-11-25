// RUN: heir-opt --implement-linear-transforms %s | FileCheck %s

// On a backend with a native linear transform, each (output, input) block
// becomes one kernel.linear_transform that reads its rows of the shared
// diagonals operand.

// CHECK: @block
// CHECK-SAME: (%[[X:.*]]: tensor<2x8xf32>, %[[D:.*]]: tensor<4x8xf32>)
// CHECK-NOT: kernel.block_linear_transform
// CHECK-DAG: %[[X0:.*]] = tensor.extract_slice %[[X]][0, 0] [1, 8] [1, 1]
// CHECK-DAG: %[[X1:.*]] = tensor.extract_slice %[[X]][1, 0] [1, 8] [1, 1]
// CHECK-DAG: %[[A:.*]] = kernel.linear_transform %[[X0]], %[[D]] <diagonal_indices = [1], source_row_indices = [0]>
// CHECK-DAG: %[[B:.*]] = kernel.linear_transform %[[X0]], %[[D]] <diagonal_indices = [0, 1], source_row_indices = [1, 2]>
// CHECK-DAG: %[[C:.*]] = kernel.linear_transform %[[X1]], %[[D]] <diagonal_indices = [2], source_row_indices = [3]>
// CHECK: %[[SUM:.*]] = arith.addf %[[B]], %[[C]]
// CHECK: tensor.insert_slice %[[A]]
// CHECK: tensor.insert_slice %[[SUM]]
module attributes {backend.lattigo, scheme.ckks} {
  func.func @block(%x: tensor<2x8xf32>, %diagonals: tensor<4x8xf32>) -> tensor<2x8xf32> {
    %0 = kernel.block_linear_transform %x, %diagonals <
      output_ciphertexts = [0, 1, 1, 1],
      input_ciphertexts = [0, 0, 0, 1],
      shifts = [1, 0, 1, 2]
    > : tensor<2x8xf32>, tensor<4x8xf32> -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }
}
