// RUN: not heir-opt --strided-layout-propagation --split-input-file %s 2>&1 | FileCheck %s

// A product of two secret tensors is not linear in one of them.
// CHECK: 'linalg.matmul' op has more than one secret input
func.func @two_secret_inputs(%a: !secret.secret<tensor<4x4xf32>>, %b: !secret.secret<tensor<4x4xf32>>) -> !secret.secret<tensor<4x4xf32>> {
  %init = arith.constant dense<0.0> : tensor<4x4xf32>
  %0 = secret.generic(%a: !secret.secret<tensor<4x4xf32>>, %b: !secret.secret<tensor<4x4xf32>>) {
  ^body(%x: tensor<4x4xf32>, %y: tensor<4x4xf32>):
    %1 = linalg.matmul ins(%x, %y : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    secret.yield %1 : tensor<4x4xf32>
  } -> !secret.secret<tensor<4x4xf32>>
  return %0 : !secret.secret<tensor<4x4xf32>>
}

// -----

// A maximum over a window is not linear.
// CHECK: 'linalg.pooling_nchw_max' op is not a linear operation in its secret operand
func.func @max_pool(%a: !secret.secret<tensor<1x1x4x4xf32>>) -> !secret.secret<tensor<1x1x2x2xf32>> {
  %init = arith.constant dense<0.0> : tensor<1x1x2x2xf32>
  %window = tensor.empty() : tensor<2x2xf32>
  %0 = secret.generic(%a: !secret.secret<tensor<1x1x4x4xf32>>) {
  ^body(%x: tensor<1x1x4x4xf32>):
    %1 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%x, %window : tensor<1x1x4x4xf32>, tensor<2x2xf32>) outs(%init : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
    secret.yield %1 : tensor<1x1x2x2xf32>
  } -> !secret.secret<tensor<1x1x2x2xf32>>
  return %0 : !secret.secret<tensor<1x1x2x2xf32>>
}
