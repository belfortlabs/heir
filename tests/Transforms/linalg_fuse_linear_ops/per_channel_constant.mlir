// RUN: heir-opt --linalg-fuse-linear-ops %s | FileCheck %s

// A per-channel scale that was already broadcast to the conv's result shape
// and folded into a constant (as linalg-canonicalizations does for a
// BatchNorm) fuses into the filter like the unbroadcast rank-1 scale would.

// CHECK: func.func @fuse_folded_per_channel_scale
// CHECK: %[[SCALE:.*]] = arith.constant dense<{{.*}}2.000000e+00{{.*}}3.000000e+00{{.*}}> : tensor<2x1x2x2xf32>
// CHECK: %[[SCALED_W:.*]] = arith.mulf %arg1, %[[SCALE]]
// CHECK: %[[RESULT:.*]] = linalg.conv_2d_nchw_fchw
// CHECK-SAME: ins(%arg0, %[[SCALED_W]]
// CHECK-NOT: arith.mulf
// CHECK: return %[[RESULT]]
func.func @fuse_folded_per_channel_scale(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<2x1x2x2xf32>) -> tensor<1x2x2x2xf32> {
  %zero = arith.constant dense<0.000000e+00> : tensor<1x2x2x2xf32>
  %scale = arith.constant dense<[[[[2.0, 2.0], [2.0, 2.0]], [[3.0, 3.0], [3.0, 3.0]]]]> : tensor<1x2x2x2xf32>
  %0 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x1x3x3xf32>, tensor<2x1x2x2xf32>) outs(%zero : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %1 = arith.mulf %0, %scale : tensor<1x2x2x2xf32>
  return %1 : tensor<1x2x2x2xf32>
}

// A constant that varies along a spatial dimension is not a per-channel scale.

// CHECK: func.func @no_fuse_spatial_scale
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-SAME: ins(%arg0, %arg1
// CHECK: arith.mulf
func.func @no_fuse_spatial_scale(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<2x1x2x2xf32>) -> tensor<1x2x2x2xf32> {
  %zero = arith.constant dense<0.000000e+00> : tensor<1x2x2x2xf32>
  %scale = arith.constant dense<[[[[2.0, 3.0], [2.0, 3.0]], [[2.0, 3.0], [2.0, 3.0]]]]> : tensor<1x2x2x2xf32>
  %0 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x1x3x3xf32>, tensor<2x1x2x2xf32>) outs(%zero : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %1 = arith.mulf %0, %scale : tensor<1x2x2x2xf32>
  return %1 : tensor<1x2x2x2xf32>
}

// A conv whose result has another user is not fused, since that user would
// see the scaled result (as in a HerPN `x * x + a1 * x`).

// CHECK: func.func @no_fuse_multiple_uses
// CHECK: %[[CONV:.*]] = linalg.conv_2d_nchw_fchw
// CHECK-SAME: ins(%arg0, %arg1
// CHECK: arith.mulf %[[CONV]], %[[CONV]]
// CHECK: arith.mulf %[[CONV]], %{{.*}}
func.func @no_fuse_multiple_uses(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<2x1x2x2xf32>) -> tensor<1x2x2x2xf32> {
  %zero = arith.constant dense<0.000000e+00> : tensor<1x2x2x2xf32>
  %scale = arith.constant dense<[[[[2.0, 2.0], [2.0, 2.0]], [[3.0, 3.0], [3.0, 3.0]]]]> : tensor<1x2x2x2xf32>
  %0 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x1x3x3xf32>, tensor<2x1x2x2xf32>) outs(%zero : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %1 = arith.mulf %0, %0 : tensor<1x2x2x2xf32>
  %2 = arith.mulf %0, %scale : tensor<1x2x2x2xf32>
  %3 = arith.addf %1, %2 : tensor<1x2x2x2xf32>
  return %3 : tensor<1x2x2x2xf32>
}

// Resource-constant weights stay a resource constant after fusing a constant
// scale, so that assign_layout can still fold them into packed constants.

// CHECK: func.func @fuse_into_resource_weights
// CHECK: %[[W:.*]] = arith.constant dense_resource<fused_weights{{.*}}> : tensor<2x1x2x2xf32>
// CHECK-NOT: arith.mulf
// CHECK: %[[RESULT:.*]] = linalg.conv_2d_nchw_fchw
// CHECK-SAME: ins(%arg0, %[[W]]
// CHECK-NOT: arith.mulf
// CHECK: return %[[RESULT]]
// CHECK: fused_weights{{.*}}: "0x040000000000004000000040000000400000004000004040000040400000404000004040"
func.func @fuse_into_resource_weights(%arg0: tensor<1x1x3x3xf32>) -> tensor<1x2x2x2xf32> {
  %weights = arith.constant dense_resource<ones> : tensor<2x1x2x2xf32>
  %zero = arith.constant dense<0.000000e+00> : tensor<1x2x2x2xf32>
  %scale = arith.constant dense<[[[[2.0, 2.0], [2.0, 2.0]], [[3.0, 3.0], [3.0, 3.0]]]]> : tensor<1x2x2x2xf32>
  %0 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %weights : tensor<1x1x3x3xf32>, tensor<2x1x2x2xf32>) outs(%zero : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %1 = arith.mulf %0, %scale : tensor<1x2x2x2xf32>
  return %1 : tensor<1x2x2x2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      ones: "0x040000000000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F"
    }
  }
#-}
