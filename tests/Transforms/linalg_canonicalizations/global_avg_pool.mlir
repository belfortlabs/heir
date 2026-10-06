// RUN: heir-opt --linalg-canonicalizations %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// A global average pool, flattened into a linear layer, as torch exports it.
// The pool and the matmul become one matmul over the flattened input, with
// each weight repeated over the pooled window and divided by its size.

// CHECK: func.func @global_avg_pool_linear
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x2x2x2xf32>)
// CHECK-NOT: linalg.pooling_nchw_sum
// CHECK-NOT: linalg.conv_2d_nchw_fchw
// CHECK-DAG: %[[FLAT:.*]] = tensor.collapse_shape %[[ARG0]] {{\[\[}}0], [1, 2, 3]] : tensor<1x2x2x2xf32> into tensor<1x8xf32>
// CHECK-DAG: %[[W:.*]] = arith.constant dense_resource<folded_pool_weights> : tensor<2x8xf32>
// CHECK: %[[T:.*]] = linalg.transpose ins(%[[W]] : tensor<2x8xf32>) outs(%{{.*}} : tensor<8x2xf32>) permutation = [1, 0]
// CHECK: %[[RES:.*]] = linalg.matmul ins(%[[FLAT]], %[[T]] : tensor<1x8xf32>, tensor<8x2xf32>)
// CHECK: return %[[RES]]
func.func @global_avg_pool_linear(%arg0: tensor<1x2x2x2xf32>) -> tensor<1x2xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 4.000000e+00 : f32
  %weights = arith.constant dense_resource<weights> : tensor<2x2xf32>
  %0 = tensor.empty() : tensor<1x2x1x1xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x2x1x1xf32>) -> tensor<1x2x1x1xf32>
  %2 = tensor.empty() : tensor<2x2xf32>
  %3 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %2 : tensor<1x2x2x2xf32>, tensor<2x2xf32>) outs(%1 : tensor<1x2x1x1xf32>) -> tensor<1x2x1x1xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<1x2x1x1xf32>) outs(%0 : tensor<1x2x1x1xf32>) {
  ^bb0(%in: f32, %out: f32):
    %9 = arith.divf %in, %cst_0 : f32
    linalg.yield %9 : f32
  } -> tensor<1x2x1x1xf32>
  %collapsed = tensor.collapse_shape %4 [[0], [1, 2, 3]] : tensor<1x2x1x1xf32> into tensor<1x2xf32>
  %5 = tensor.empty() : tensor<2x2xf32>
  %transposed = linalg.transpose ins(%weights : tensor<2x2xf32>) outs(%5 : tensor<2x2xf32>) permutation = [1, 0]
  %6 = tensor.empty() : tensor<1x2xf32>
  %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<1x2xf32>) -> tensor<1x2xf32>
  %8 = linalg.matmul ins(%collapsed, %transposed : tensor<1x2xf32>, tensor<2x2xf32>) outs(%7 : tensor<1x2xf32>) -> tensor<1x2xf32>
  return %8 : tensor<1x2xf32>
}

// The same with the division already on tensors and dense, untransposed
// weights. The matmul keeps its bias.

// CHECK: func.func @global_avg_pool_linear_dense
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x2x1x2xf32>)
// CHECK-DAG: %[[BIAS:.*]] = arith.constant dense<1.000000e+00> : tensor<1x3xf32>
// CHECK-DAG: %[[W:.*]] = arith.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<4x3xf32>
// CHECK-DAG: %[[FLAT:.*]] = tensor.collapse_shape %[[ARG0]] {{\[\[}}0], [1, 2, 3]] : tensor<1x2x1x2xf32> into tensor<1x4xf32>
// CHECK: %[[RES:.*]] = linalg.matmul ins(%[[FLAT]], %[[W]] : tensor<1x4xf32>, tensor<4x3xf32>) outs(%[[BIAS]] : tensor<1x3xf32>)
// CHECK: return %[[RES]]
func.func @global_avg_pool_linear_dense(%arg0: tensor<1x2x1x2xf32>) -> tensor<1x3xf32> {
  %zero = arith.constant dense<0.000000e+00> : tensor<1x2x1x1xf32>
  %size = arith.constant dense<2.000000e+00> : tensor<1x2x1x1xf32>
  %window = tensor.empty() : tensor<1x2xf32>
  %weights = arith.constant dense<[[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]> : tensor<2x3xf32>
  %bias = arith.constant dense<1.000000e+00> : tensor<1x3xf32>
  %0 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %window : tensor<1x2x1x2xf32>, tensor<1x2xf32>) outs(%zero : tensor<1x2x1x1xf32>) -> tensor<1x2x1x1xf32>
  %1 = arith.divf %0, %size : tensor<1x2x1x1xf32>
  %collapsed = tensor.collapse_shape %1 [[0], [1, 2, 3]] : tensor<1x2x1x1xf32> into tensor<1x2xf32>
  %2 = linalg.matmul ins(%collapsed, %weights : tensor<1x2xf32>, tensor<2x3xf32>) outs(%bias : tensor<1x3xf32>) -> tensor<1x3xf32>
  return %2 : tensor<1x3xf32>
}

// A pool over part of the input is left to the average-pool rewrite.

// CHECK: func.func @partial_pool
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK: linalg.matmul
// CHECK-SAME: tensor<1x8xf32>, tensor<8x3xf32>
func.func @partial_pool(%arg0: tensor<1x2x4x4xf32>) -> tensor<1x3xf32> {
  %zero = arith.constant dense<0.000000e+00> : tensor<1x2x2x2xf32>
  %window = tensor.empty() : tensor<2x2xf32>
  %weights = arith.constant dense<1.000000e+00> : tensor<8x3xf32>
  %bias = arith.constant dense<0.000000e+00> : tensor<1x3xf32>
  %0 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %window : tensor<1x2x4x4xf32>, tensor<2x2xf32>) outs(%zero : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %collapsed = tensor.collapse_shape %0 [[0], [1, 2, 3]] : tensor<1x2x2x2xf32> into tensor<1x8xf32>
  %1 = linalg.matmul ins(%collapsed, %weights : tensor<1x8xf32>, tensor<8x3xf32>) outs(%bias : tensor<1x3xf32>) -> tensor<1x3xf32>
  return %1 : tensor<1x3xf32>
}

// A pool whose result has another user is not folded.

// CHECK: func.func @pool_with_other_user
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK: linalg.matmul
// CHECK-SAME: tensor<1x2xf32>, tensor<2x3xf32>
func.func @pool_with_other_user(%arg0: tensor<1x2x2x2xf32>) -> (tensor<1x3xf32>, tensor<1x2x1x1xf32>) {
  %zero = arith.constant dense<0.000000e+00> : tensor<1x2x1x1xf32>
  %window = tensor.empty() : tensor<2x2xf32>
  %weights = arith.constant dense<1.000000e+00> : tensor<2x3xf32>
  %bias = arith.constant dense<0.000000e+00> : tensor<1x3xf32>
  %0 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %window : tensor<1x2x2x2xf32>, tensor<2x2xf32>) outs(%zero : tensor<1x2x1x1xf32>) -> tensor<1x2x1x1xf32>
  %collapsed = tensor.collapse_shape %0 [[0], [1, 2, 3]] : tensor<1x2x1x1xf32> into tensor<1x2xf32>
  %1 = linalg.matmul ins(%collapsed, %weights : tensor<1x2xf32>, tensor<2x3xf32>) outs(%bias : tensor<1x3xf32>) -> tensor<1x3xf32>
  return %1, %0 : tensor<1x3xf32>, tensor<1x2x1x1xf32>
}

// CHECK: folded_pool_weights: "0x040000000000803E0000803E0000803E0000803E0000003F0000003F0000003F0000003F0000403F0000403F0000403F0000403F0000803F0000803F0000803F0000803F"

{-#
  dialect_resources: {
    builtin: {
      weights: "0x040000000000803F000000400000404000008040"
    }
  }
#-}
