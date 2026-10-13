// RUN: heir-opt --linalg-canonicalizations --layout-propagation %s | FileCheck %s

// A depthwise conv after a zero pad on the gapped result of a strided conv is
// rewritten to a regular conv, which folds the pad into its read of the gapped
// data.

// CHECK: @strided_then_padded_depthwise
func.func @strided_then_padded_depthwise(%arg0: !secret.secret<tensor<1x1x8x8xf32>>) -> !secret.secret<tensor<1x4x4x4xf32>> {
  %filter = arith.constant dense<2.500000e-01> : tensor<4x1x2x2xf32>
  %filter2 = arith.constant dense<[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0], [-7.0, -8.0, -9.0]], [[0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]], [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]> : tensor<4x3x3xf32>
  %init = arith.constant dense<0.000000e+00> : tensor<1x4x4x4xf32>
  %zero = arith.constant 0.000000e+00 : f32

  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x8x8xf32>>) {
  ^body(%input0: tensor<1x1x8x8xf32>):
    // CHECK: %[[strided:[^ ]+]] = linalg.conv_2d_nchw_fchw
    // CHECK-SAME: strides = dense<2>
    // CHECK-NOT: tensor_ext.convert_layout
    // CHECK: tensor.pad %[[strided]]
    // CHECK-NOT: tensor_ext.convert_layout
    // CHECK: linalg.conv_2d_nchw_fchw
    // CHECK-SAME: heir.conv_folded_padding = 1
    // CHECK-SAME: gap_factor = 2
    // CHECK-NOT: linalg.depthwise_conv_2d_nchw_chw
    %1 = linalg.conv_2d_nchw_fchw
      { dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
      ins(%input0, %filter : tensor<1x1x8x8xf32>, tensor<4x1x2x2xf32>)
      outs(%init : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    %padded = tensor.pad %1 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%i: index, %j: index, %k: index, %l: index):
      tensor.yield %zero : f32
    } : tensor<1x4x4x4xf32> to tensor<1x4x6x6xf32>
    %2 = linalg.depthwise_conv_2d_nchw_chw
      { dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
      ins(%padded, %filter2 : tensor<1x4x6x6xf32>, tensor<4x3x3xf32>)
      outs(%init : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    secret.yield %2 : tensor<1x4x4x4xf32>
  } -> !secret.secret<tensor<1x4x4x4xf32>>
  return %0 : !secret.secret<tensor<1x4x4x4xf32>>
}
