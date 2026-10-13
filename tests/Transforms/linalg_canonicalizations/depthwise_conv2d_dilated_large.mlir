// RUN: heir-opt --linalg-canonicalizations --mlir-elide-resource-strings-if-larger=16 %s | FileCheck %s

// The expanded 256x256x5x5 filter exceeds the 1 MiB cap of the undilate
// pattern, so the depthwise pattern must apply the dilation itself.
// CHECK: func.func @dilated_large_filter
// CHECK: %[[filter:.*]] = arith.constant dense_resource<depthwise_filter> : tensor<256x256x5x5xf32>
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-SAME: dilations = dense<1>
// CHECK-SAME: tensor<1x256x8x8xf32>, tensor<256x256x5x5xf32>
// CHECK-NOT: linalg.depthwise_conv_2d_nchw_chw
func.func @dilated_large_filter(%arg0: tensor<1x256x8x8xf32>) -> tensor<1x256x4x4xf32> {
  %init = arith.constant dense<0.000000e+00> : tensor<1x256x4x4xf32>
  %filter = arith.constant dense<1.0> : tensor<256x3x3xf32>
  %0 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<2> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %filter : tensor<1x256x8x8xf32>, tensor<256x3x3xf32>) outs(%init : tensor<1x256x4x4xf32>) -> tensor<1x256x4x4xf32>
  return %0 : tensor<1x256x4x4xf32>
}
