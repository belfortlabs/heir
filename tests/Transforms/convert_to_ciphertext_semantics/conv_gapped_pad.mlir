// RUN: heir-opt --layout-propagation=min-slot-count=64 --convert-to-ciphertext-semantics=min-slot-count=64 %s | FileCheck %s

// The conv after a zero pad on the gapped result of a strided conv folds the
// pad, so its matrix only holds the diagonals its filter taps produce.

module attributes {backend.openfhe, backend.config_override = {has_kernel_linear_transform = true}} {
// The conv after the pad reads the gapped data in place. A tap's slot distance
// depends on its channel's position in the 2x2 block, so the 4 * 3 * 3 taps
// fall on 49 diagonals.
// CHECK: func.func @strided_then_padded_conv(
// CHECK: tensor_ext.rotate_and_reduce
// CHECK: tensor_ext.rotate_and_reduce
// CHECK-SAME: tensor_ext.diagonal_indices = array<i32: 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63>
func.func @strided_then_padded_conv(%arg0: !secret.secret<tensor<1x1x8x8xf32>>) -> !secret.secret<tensor<1x4x4x4xf32>> {
  %filter = arith.constant dense<2.500000e-01> : tensor<4x1x2x2xf32>
  %filter2 = arith.constant dense<1.000000e-01> : tensor<4x4x3x3xf32>
  %init = arith.constant dense<0.000000e+00> : tensor<1x4x4x4xf32>
  %zero = arith.constant 0.000000e+00 : f32
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x8x8xf32>>) {
  ^body(%input0: tensor<1x1x8x8xf32>):
    %1 = linalg.conv_2d_nchw_fchw { dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> } ins(%input0, %filter : tensor<1x1x8x8xf32>, tensor<4x1x2x2xf32>) outs(%init : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    %padded = tensor.pad %1 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%i: index, %j: index, %k: index, %l: index):
      tensor.yield %zero : f32
    } : tensor<1x4x4x4xf32> to tensor<1x4x6x6xf32>
    %2 = linalg.conv_2d_nchw_fchw { dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> } ins(%padded, %filter2 : tensor<1x4x6x6xf32>, tensor<4x4x3x3xf32>) outs(%init : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    secret.yield %2 : tensor<1x4x4x4xf32>
  } -> !secret.secret<tensor<1x4x4x4xf32>>
  return %0 : !secret.secret<tensor<1x4x4x4xf32>>
}
// The same for a 1-D conv: 4 * 3 = 12 diagonals.
// CHECK: func.func @strided_then_padded_conv1d(
// CHECK: tensor_ext.rotate_and_reduce
// CHECK: tensor_ext.rotate_and_reduce
// CHECK-SAME: tensor_ext.diagonal_indices = array<i32: 0, 1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31>
func.func @strided_then_padded_conv1d(%arg0: !secret.secret<tensor<1x1x16xf32>>) -> !secret.secret<tensor<1x4x8xf32>> {
  %filter = arith.constant dense<2.500000e-01> : tensor<4x1x2xf32>
  %filter2 = arith.constant dense<1.000000e-01> : tensor<4x4x3xf32>
  %init = arith.constant dense<0.000000e+00> : tensor<1x4x8xf32>
  %zero = arith.constant 0.000000e+00 : f32
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x16xf32>>) {
  ^body(%input0: tensor<1x1x16xf32>):
    %1 = linalg.conv_1d_ncw_fcw { dilations = dense<1> : tensor<1xi64>, strides = dense<2> : tensor<1xi64> } ins(%input0, %filter : tensor<1x1x16xf32>, tensor<4x1x2xf32>) outs(%init : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %padded = tensor.pad %1 low[0, 0, 1] high[0, 0, 1] {
    ^bb0(%i: index, %j: index, %k: index):
      tensor.yield %zero : f32
    } : tensor<1x4x8xf32> to tensor<1x4x10xf32>
    %2 = linalg.conv_1d_ncw_fcw { dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64> } ins(%padded, %filter2 : tensor<1x4x10xf32>, tensor<4x4x3xf32>) outs(%init : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    secret.yield %2 : tensor<1x4x8xf32>
  } -> !secret.secret<tensor<1x4x8xf32>>
  return %0 : !secret.secret<tensor<1x4x8xf32>>
}
}
