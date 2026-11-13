// RUN: heir-opt --layout-propagation %s | FileCheck %s

// A vecmat on the gapped result of a strided conv produces a row-major
// result, so the conv that follows must see kernel info with gap factor 1.
// Runs without linalg-canonicalizations, which would rewrite the vecmat as a
// matvec.

// CHECK: @strided_conv_vecmat_conv
// CHECK: linalg.vecmat
// CHECK-SAME: heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 4>}
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-SAME: heir.kernel_info = {gap_factor = 1 : i64, input_shape = array<i64: 1, 4, 1, 1>, result_shape = array<i64: 1, 2, 1, 1>}
func.func @strided_conv_vecmat_conv(%arg0: !secret.secret<tensor<1x1x8x8xf32>>) -> !secret.secret<tensor<1x2x1x1xf32>> {
  %filter = arith.constant dense<2.500000e-01> : tensor<4x1x2x2xf32>
  %bias = arith.constant dense<1.000000e+00> : tensor<1x4x4x4xf32>
  %matrix = arith.constant dense<6.250000e-02> : tensor<64x4xf32>
  %init = arith.constant dense<0.000000e+00> : tensor<4xf32>
  %fc = arith.constant dense<5.000000e-01> : tensor<2x4x1x1xf32>
  %fc_bias = arith.constant dense<1.000000e+00> : tensor<1x2x1x1xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x8x8xf32>>) {
  ^body(%input0: tensor<1x1x8x8xf32>):
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%input0, %filter : tensor<1x1x8x8xf32>, tensor<4x1x2x2xf32>) outs(%bias : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    %2 = tensor.collapse_shape %1 [[0, 1, 2, 3]] : tensor<1x4x4x4xf32> into tensor<64xf32>
    %3 = linalg.vecmat ins(%2, %matrix : tensor<64xf32>, tensor<64x4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
    %4 = tensor.expand_shape %3 [[0, 1, 2, 3]] output_shape [1, 4, 1, 1] : tensor<4xf32> into tensor<1x4x1x1xf32>
    %5 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%4, %fc : tensor<1x4x1x1xf32>, tensor<2x4x1x1xf32>) outs(%fc_bias : tensor<1x2x1x1xf32>) -> tensor<1x2x1x1xf32>
    secret.yield %5 : tensor<1x2x1x1xf32>
  } -> !secret.secret<tensor<1x2x1x1xf32>>
  return %0 : !secret.secret<tensor<1x2x1x1xf32>>
}
