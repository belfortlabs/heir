// RUN: not heir-opt --layout-propagation --split-input-file %s 2>&1 | FileCheck %s

// Operations that are linear in one secret tensor have their kernel in the
// strided path.
// CHECK: 'linalg.matvec' op is only supported with strided layouts
func.func @matvec(%arg0: !secret.secret<tensor<16xf32>>, %m: tensor<16x16xf32>) -> !secret.secret<tensor<16xf32>> {
  %init = arith.constant dense<0.000000e+00> : tensor<16xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<16xf32>>) {
  ^body(%input0: tensor<16xf32>):
    %1 = linalg.matvec ins(%m, %input0 : tensor<16x16xf32>, tensor<16xf32>) outs(%init : tensor<16xf32>) -> tensor<16xf32>
    secret.yield %1 : tensor<16xf32>
  } -> !secret.secret<tensor<16xf32>>
  return %0 : !secret.secret<tensor<16xf32>>
}

// -----

// The bicyclic kernels need pairwise coprime dimensions.
// CHECK: 'linalg.matmul' op has no kernel for these operands
func.func @matmul_non_coprime(%arg0: !secret.secret<tensor<4x6xf32>>, %arg1: !secret.secret<tensor<6x9xf32>>) -> !secret.secret<tensor<4x9xf32>> {
  %cst = arith.constant dense<0.000000e+00> : tensor<4x9xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<4x6xf32>>, %arg1: !secret.secret<tensor<6x9xf32>>) {
  ^body(%input0: tensor<4x6xf32>, %input1: tensor<6x9xf32>):
    %1 = linalg.matmul ins(%input0, %input1 : tensor<4x6xf32>, tensor<6x9xf32>) outs(%cst : tensor<4x9xf32>) -> tensor<4x9xf32>
    secret.yield %1 : tensor<4x9xf32>
  } -> !secret.secret<tensor<4x9xf32>>
  return %0 : !secret.secret<tensor<4x9xf32>>
}
