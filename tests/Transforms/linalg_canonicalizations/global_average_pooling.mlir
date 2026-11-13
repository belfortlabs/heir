// RUN: heir-opt --linalg-canonicalizations %s | FileCheck %s

// A global average pool becomes a matvec on the flattened input

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  // CHECK: func.func @global_avg_pool
  // CHECK-SAME: (%[[arg0:.*]]: tensor<1x2x2x2xf32>)
  // CHECK-DAG: %[[out:.*]] = arith.constant dense<0.0{{.*}}> : tensor<2xf32>
  // CHECK-DAG: %[[matrix:.*]] = arith.constant dense<{{\[\[}}2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01]]> : tensor<2x8xf32>
  // CHECK: %[[flat:.*]] = tensor.collapse_shape %[[arg0]] {{\[\[}}0, 1, 2, 3]] : tensor<1x2x2x2xf32> into tensor<8xf32>
  // CHECK: %[[matvec:.*]] = linalg.matvec ins(%[[matrix]], %[[flat]] : tensor<2x8xf32>, tensor<8xf32>) outs(%[[out]] : tensor<2xf32>)
  // CHECK: %[[result:.*]] = tensor.expand_shape %[[matvec]] {{\[\[}}0, 1, 2, 3]] output_shape [1, 2, 1, 1] : tensor<2xf32> into tensor<1x2x1x1xf32>
  // CHECK-NOT: linalg.conv_2d_nchw_fchw
  // CHECK: return %[[result]]
  func.func @global_avg_pool(%arg0: tensor<1x2x2x2xf32>) -> tensor<1x2x1x1xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 4.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x2x1x1xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x2x1x1xf32>) -> tensor<1x2x1x1xf32>
    %2 = tensor.empty() : tensor<2x2xf32>
    %3 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %2 : tensor<1x2x2x2xf32>, tensor<2x2xf32>) outs(%1 : tensor<1x2x1x1xf32>) -> tensor<1x2x1x1xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<1x2x1x1xf32>) outs(%0 : tensor<1x2x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.divf %in, %cst_1 : f32
      linalg.yield %5 : f32
    } -> tensor<1x2x1x1xf32>
    return %4 : tensor<1x2x1x1xf32>
  }
}
