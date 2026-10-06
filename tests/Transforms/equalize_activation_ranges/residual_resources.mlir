// RUN: heir-opt --equalize-activation-ranges %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// A residual block as torch-mlir exports it. The skip connection puts both
// ReLUs in one class. The second domain [-8, 8] needs the larger factor
// 2 * 4 / 0.5 = 16. conv1 moves from factor 1 to 16, conv2 stays at 16 and
// only its bias is divided, and the final matmul moves back to factor 1.
// Rescaled weights and biases stay resources.

// CHECK: func.func @residual
// CHECK-DAG: %[[W1:.*]] = arith.constant dense_resource<[[W1_NAME:range_equalized.*]]> : tensor<1x1x1x1xf32>
// CHECK-DAG: %[[B1:.*]] = arith.constant dense_resource<[[B1_NAME:range_equalized.*]]> : tensor<1xf32>
// CHECK-DAG: %[[W2:.*]] = arith.constant dense_resource<w2> : tensor<1x1x3x3xf32>
// CHECK-DAG: %[[B2:.*]] = arith.constant dense_resource<[[B2_NAME:range_equalized.*]]> : tensor<1xf32>
// CHECK-DAG: %[[W3:.*]] = arith.constant dense_resource<[[W3_NAME:range_equalized.*]]> : tensor<4x1xf32>
// CHECK: linalg.broadcast ins(%[[B1]]
// CHECK: linalg.conv_2d_nchw_fchw {{.*}} ins(%{{.*}}, %[[W1]] :
// CHECK: domain_lower = -2.500000e-01 : f64, domain_upper = 2.500000e-01 : f64
// CHECK: linalg.broadcast ins(%[[B2]]
// CHECK: linalg.conv_2d_nchw_fchw {{.*}} ins(%{{.*}}, %[[W2]] :
// CHECK: domain_lower = -5.000000e-01 : f64, domain_upper = 5.000000e-01 : f64
// CHECK: linalg.matmul ins(%{{.*}}, %[[W3]] :
// CHECK-DAG: [[W1_NAME]]: "0x040000000000803F"
// CHECK-DAG: [[B1_NAME]]: "0x0400000000000040"
// CHECK-DAG: [[B2_NAME]]: "0x0400000000004040"
// CHECK-DAG: [[W3_NAME]]: "0x0400000000000041000000410000004100000041"
func.func @residual(%arg0: tensor<1x1x2x2xf32> {secret.secret}) -> tensor<1x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %w1 = arith.constant dense_resource<w1> : tensor<1x1x1x1xf32>
  %b1 = arith.constant dense_resource<b1> : tensor<1xf32>
  %w2 = arith.constant dense_resource<w2> : tensor<1x1x3x3xf32>
  %b2 = arith.constant dense_resource<b2> : tensor<1xf32>
  %w3 = arith.constant dense_resource<w3> : tensor<4x1xf32>
  %empty = tensor.empty() : tensor<1x1x2x2xf32>
  %bias1 = linalg.broadcast ins(%b1 : tensor<1xf32>) outs(%empty : tensor<1x1x2x2xf32>) dimensions = [0, 2, 3]
  %0 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %w1 : tensor<1x1x2x2xf32>, tensor<1x1x1x1xf32>) outs(%bias1 : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0 : tensor<1x1x2x2xf32>) outs(%empty : tensor<1x1x2x2xf32>) attrs = {domain_lower = -4.0 : f64, domain_upper = 4.0 : f64} {
  ^bb0(%in: f32, %out: f32):
    %8 = arith.cmpf ugt, %in, %cst : f32
    %9 = arith.select %8, %in, %cst : f32
    linalg.yield %9 : f32
  } -> tensor<1x1x2x2xf32>
  %padded = tensor.pad %1 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
    tensor.yield %cst : f32
  } : tensor<1x1x2x2xf32> to tensor<1x1x4x4xf32>
  %bias2 = linalg.broadcast ins(%b2 : tensor<1xf32>) outs(%empty : tensor<1x1x2x2xf32>) dimensions = [0, 2, 3]
  %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded, %w2 : tensor<1x1x4x4xf32>, tensor<1x1x3x3xf32>) outs(%bias2 : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %1 : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>) outs(%empty : tensor<1x1x2x2xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %8 = arith.addf %in, %in_0 : f32
    linalg.yield %8 : f32
  } -> tensor<1x1x2x2xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<1x1x2x2xf32>) outs(%empty : tensor<1x1x2x2xf32>) attrs = {domain_lower = -8.0 : f64, domain_upper = 8.0 : f64} {
  ^bb0(%in: f32, %out: f32):
    %8 = arith.cmpf ugt, %in, %cst : f32
    %9 = arith.select %8, %in, %cst : f32
    linalg.yield %9 : f32
  } -> tensor<1x1x2x2xf32>
  %collapsed = tensor.collapse_shape %4 [[0], [1, 2, 3]] : tensor<1x1x2x2xf32> into tensor<1x4xf32>
  %empty_out = tensor.empty() : tensor<1x1xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty_out : tensor<1x1xf32>) -> tensor<1x1xf32>
  %5 = linalg.matmul ins(%collapsed, %w3 : tensor<1x4xf32>, tensor<4x1xf32>) outs(%fill : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %5 : tensor<1x1xf32>
}

{-#
  dialect_resources: {
    builtin: {
      w1: "0x0400000000008041",
      b1: "0x0400000000000042",
      w2: "0x040000000000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F",
      b2: "0x0400000000004042",
      w3: "0x040000000000003F0000003F0000003F0000003F"
    }
  }
#-}
