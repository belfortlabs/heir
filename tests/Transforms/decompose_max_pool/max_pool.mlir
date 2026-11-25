// RUN: heir-opt --decompose-max-pool %s | FileCheck %s

// A 2x2 window with stride 2 and one row and column of padding in front. The
// tap at window position (1, 1) never reads padding; the other three taps
// are replaced by it where they do.

// CHECK: func.func @max_pool
// CHECK-SAME: (%[[x:.*]]: tensor<1x1x3x3xf32>)
// CHECK: %[[zero:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[padded:.*]] = tensor.pad %[[x]] low[0, 0, 1, 1] high[0, 0, 0, 0]
// CHECK: tensor.yield %[[zero]] : f32
// CHECK: %[[anchor:.*]] = linalg.generic
// CHECK-SAME: ins(%[[padded]] :
// CHECK: %[[tap00:.*]] = linalg.generic
// CHECK: %[[mask:.*]] = linalg.broadcast
// CHECK: %[[stand_in:.*]] = arith.mulf %[[anchor]], %[[mask]]
// CHECK: %[[t00:.*]] = arith.addf %[[tap00]], %[[stand_in]]
// CHECK: %[[zeros:.*]] = arith.constant dense<0.000000e+00> : tensor<1x1x2x2xf32>
// CHECK: %[[diff:.*]] = arith.subf %[[t00]],
// CHECK: %[[relu:.*]] = arith.maximumf %[[diff]], %[[zeros]] {domain_lower = -4.000000e+00 : f64, domain_upper = 4.000000e+00 : f64}
// CHECK: arith.addf %{{.*}}, %[[relu]]
// CHECK-NOT: linalg.pooling_nchw_max
// CHECK: return
func.func @max_pool(%x: tensor<1x1x3x3xf32>) -> tensor<1x1x2x2xf32> {
  %neg_inf = arith.constant 0xFF800000 : f32
  %padded = tensor.pad %x low[0, 0, 1, 1] high[0, 0, 0, 0] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %neg_inf : f32
  } : tensor<1x1x3x3xf32> to tensor<1x1x4x4xf32>
  %empty = tensor.empty() : tensor<1x1x2x2xf32>
  %init = linalg.fill ins(%neg_inf : f32) outs(%empty : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
  %window = tensor.empty() : tensor<2x2xf32>
  %out = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, domain_lower = -1.0, domain_upper = 3.0} ins(%padded, %window : tensor<1x1x4x4xf32>, tensor<2x2xf32>) outs(%init : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
  return %out : tensor<1x1x2x2xf32>
}

// A zero pad is not the identity of the maximum: the padded tensor is read as
// it is and no tap needs a stand-in.

// CHECK: func.func @zero_padded
// CHECK: %[[padded:.*]] = tensor.pad
// CHECK-NOT: tensor.pad
// CHECK-NOT: linalg.broadcast
// CHECK: arith.maximumf
// CHECK: return
func.func @zero_padded(%x: tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32> {
  %zero = arith.constant 0.0 : f32
  %padded = tensor.pad %x low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %zero : f32
  } : tensor<1x1x2x2xf32> to tensor<1x1x4x4xf32>
  %init = tensor.empty() : tensor<1x1x2x2xf32>
  %window = tensor.empty() : tensor<2x2xf32>
  %out = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded, %window : tensor<1x1x4x4xf32>, tensor<2x2xf32>) outs(%init : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
  return %out : tensor<1x1x2x2xf32>
}
