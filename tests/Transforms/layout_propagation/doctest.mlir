// RUN: heir-opt --layout-propagation=min-slot-count=16 %s | FileCheck %s

// A product of two secret matrices. The dimensions are pairwise coprime, so
// the bicyclic kernel applies, and both operands get a bicyclic layout.

// CHECK: #secret.kernel<name = "MatmulBicyclic"
// CHECK: @matmul
// CHECK: linalg.matmul
func.func @matmul(
    %arg0: !secret.secret<tensor<3x5xf32>>,
    %arg1: !secret.secret<tensor<5x2xf32>>) -> !secret.secret<tensor<3x2xf32>> {
  %cst = arith.constant dense<0.000000e+00> : tensor<3x2xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<3x5xf32>>, %arg1: !secret.secret<tensor<5x2xf32>>) {
  ^body(%input0: tensor<3x5xf32>, %input1: tensor<5x2xf32>):
    %1 = linalg.matmul ins(%input0, %input1 : tensor<3x5xf32>, tensor<5x2xf32>) outs(%cst : tensor<3x2xf32>) -> tensor<3x2xf32>
    secret.yield %1 : tensor<3x2xf32>
  } -> !secret.secret<tensor<3x2xf32>>
  return %0 : !secret.secret<tensor<3x2xf32>>
}
