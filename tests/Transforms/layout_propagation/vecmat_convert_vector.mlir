// RUN: heir-opt --layout-propagation=min-slot-count=16 %s | FileCheck %s

// The transpose leaves the vector in a column-major layout, so layout
// propagation converts it to row major before the vecmat.

// CHECK: #kernel = #secret.kernel<name = "VecmatDiagonal", force = false>
// CHECK: @vecmat_convert_vector
func.func @vecmat_convert_vector(%arg0: !secret.secret<tensor<4x4xf32>>) -> !secret.secret<tensor<4xf32>> {
  %cst = arith.constant dense<0.000000e+00> : tensor<4xf32>
  %cst_0 = arith.constant dense<1.000000e+00> : tensor<16x4xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<4x4xf32>>) {
  ^body(%input0: tensor<4x4xf32>):
    %empty = tensor.empty() : tensor<4x4xf32>
    %t = linalg.transpose ins(%input0 : tensor<4x4xf32>) outs(%empty : tensor<4x4xf32>) permutation = [1, 0]
    // CHECK: %[[flat:.*]] = tensor.collapse_shape
    %flat = tensor.collapse_shape %t [[0, 1]] : tensor<4x4xf32> into tensor<16xf32>
    // CHECK: %[[converted:.*]] = tensor_ext.convert_layout %[[flat]]
    // CHECK: linalg.vecmat
    // CHECK-SAME: secret.kernel = #kernel
    // CHECK-SAME: ins(%[[converted]],
    %1 = linalg.vecmat ins(%flat, %cst_0 : tensor<16xf32>, tensor<16x4xf32>) outs(%cst : tensor<4xf32>) -> tensor<4xf32>
    secret.yield %1 : tensor<4xf32>
  } -> !secret.secret<tensor<4xf32>>
  return %0 : !secret.secret<tensor<4xf32>>
}
