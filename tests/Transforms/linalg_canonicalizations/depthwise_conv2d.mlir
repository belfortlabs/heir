// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

// A depthwise conv becomes a regular conv with a block-diagonal filter.
// CHECK: func.func @dense_filter
// CHECK-SAME: (%[[arg0:.*]]: tensor<1x2x4x4xf32>)
// CHECK: %[[filter:.*]] = arith.constant dense_resource<depthwise_filter> : tensor<2x2x2x2xf32>
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-SAME: dilations = dense<1>
// CHECK-SAME: strides = dense<1>
// CHECK-SAME: ins(%[[arg0]], %[[filter]] : tensor<1x2x4x4xf32>, tensor<2x2x2x2xf32>)
// CHECK-NOT: linalg.depthwise_conv_2d_nchw_chw
// CHECK: depthwise_filter: "0x040000000000803F00000040000040400000804000000000000000000000000000000000000000000000000000000000000000000000A0400000C0400000E04000000041"
func.func @dense_filter(%arg0: tensor<1x2x4x4xf32>) -> tensor<1x2x3x3xf32> {
  %init = arith.constant dense<0.000000e+00> : tensor<1x2x3x3xf32>
  %filter = arith.constant dense<[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]> : tensor<2x2x2xf32>
  %0 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %filter : tensor<1x2x4x4xf32>, tensor<2x2x2xf32>) outs(%init : tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32>
  return %0 : tensor<1x2x3x3xf32>
}

// -----

// torch-mlir collapses a Cx1xKHxKW resource filter to CxKHxKW.
// CHECK: func.func @collapsed_resource_filter
// CHECK: %[[filter:.*]] = arith.constant dense_resource<depthwise_filter> : tensor<2x2x2x2xf32>
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-SAME: tensor<1x2x4x4xf32>, tensor<2x2x2x2xf32>
// CHECK-NOT: linalg.depthwise_conv_2d_nchw_chw
// CHECK: depthwise_filter: "0x040000000000803F00000040000040400000804000000000000000000000000000000000000000000000000000000000000000000000A0400000C0400000E04000000041"
func.func @collapsed_resource_filter(%arg0: tensor<1x2x4x4xf32>) -> tensor<1x2x3x3xf32> {
  %init = arith.constant dense<0.000000e+00> : tensor<1x2x3x3xf32>
  %cst = arith.constant dense_resource<torch_tensor_2_1_2_2_torch.float32> : tensor<2x1x2x2xf32>
  %filter = tensor.collapse_shape %cst [[0, 1], [2], [3]] : tensor<2x1x2x2xf32> into tensor<2x2x2xf32>
  %0 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %filter : tensor<1x2x4x4xf32>, tensor<2x2x2xf32>) outs(%init : tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32>
  return %0 : tensor<1x2x3x3xf32>
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_2_1_2_2_torch.float32: "0x040000000000803F0000004000004040000080400000A0400000C0400000E04000000041"
    }
  }
#-}

// -----

// The strides carry over; a splat filter expands to its diagonal blocks.
// CHECK: func.func @strided_splat_filter
// CHECK: %[[filter:.*]] = arith.constant dense_resource<depthwise_filter> : tensor<2x2x2x2xf32>
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-SAME: dilations = dense<1>
// CHECK-SAME: strides = dense<2>
// CHECK-SAME: tensor<1x2x5x5xf32>, tensor<2x2x2x2xf32>
// CHECK: depthwise_filter: "0x040000000000803F0000803F0000803F0000803F00000000000000000000000000000000000000000000000000000000000000000000803F0000803F0000803F0000803F"
func.func @strided_splat_filter(%arg0: tensor<1x2x5x5xf32>) -> tensor<1x2x2x2xf32> {
  %init = arith.constant dense<0.000000e+00> : tensor<1x2x2x2xf32>
  %filter = arith.constant dense<1.0> : tensor<2x2x2xf32>
  %0 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %filter : tensor<1x2x5x5xf32>, tensor<2x2x2xf32>) outs(%init : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  return %0 : tensor<1x2x2x2xf32>
}

// -----

// A dilated depthwise conv spreads the taps in each diagonal block and becomes
// a conv with dilation 1.
// CHECK: func.func @dilated
// CHECK-NOT: arith.constant dense<{{.*}}> : tensor<2x2x3x3xf32>
// CHECK: %[[filter:.*]] = arith.constant dense_resource<depthwise_filter> : tensor<2x2x3x3xf32>
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-SAME: dilations = dense<1>
// CHECK-SAME: strides = dense<1>
// CHECK-SAME: tensor<1x2x5x5xf32>, tensor<2x2x3x3xf32>
// CHECK-NOT: linalg.depthwise_conv_2d_nchw_chw
// CHECK-NOT: arith.constant dense<{{.*}}> : tensor<2x2x3x3xf32>
// CHECK: depthwise_filter: "0x040000000000803F00000000000000400000000000000000000000000000404000000000000080400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000A040000000000000C0400000000000000000000000000000E0400000000000000041"
func.func @dilated(%arg0: tensor<1x2x5x5xf32>) -> tensor<1x2x3x3xf32> {
  %init = arith.constant dense<0.000000e+00> : tensor<1x2x3x3xf32>
  %filter = arith.constant dense<[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]> : tensor<2x2x2xf32>
  %0 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<2> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %filter : tensor<1x2x5x5xf32>, tensor<2x2x2xf32>) outs(%init : tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32>
  return %0 : tensor<1x2x3x3xf32>
}
