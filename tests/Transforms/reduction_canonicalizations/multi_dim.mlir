// RUN: heir-opt --reduction-canonicalizations --split-input-file %s | FileCheck %s

// A global average pool's sum over H and W becomes two single-dimension
// reductions, W first. The first starts from zero, the second from the
// original init.

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0)>
// CHECK: func @spatial_sum
// CHECK-DAG: %[[INIT:.*]] = arith.constant dense<1.000000e+00> : tensor<72xf32>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<72x4xf32>
// CHECK: %[[ROWS:.*]] = linalg.reduce ins(%{{.*}} : tensor<72x4x4xf32>) outs(%[[ZERO]] : tensor<72x4xf32>) dimensions = [2]
// CHECK: arith.addf
// CHECK: linalg.reduce ins(%[[ROWS]] : tensor<72x4xf32>) outs(%[[INIT]] : tensor<72xf32>) dimensions = [1]
// CHECK: arith.addf
func.func @spatial_sum(%arg0: tensor<72x4x4xf32> {secret.secret}) -> tensor<72xf32> {
  %cst = arith.constant dense<1.000000e+00> : tensor<72xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "reduction"]} ins(%arg0 : tensor<72x4x4xf32>) outs(%cst : tensor<72xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<72xf32>
  return %0 : tensor<72xf32>
}

// -----

// A product over non-adjacent dimensions starts from one.

// CHECK: func @product
// CHECK-SAME: %[[INIT:[^:]*]]: tensor<3xf32>
// CHECK: %[[ONES3:.*]] = arith.constant dense<1.000000e+00> : tensor<2x3x4xf32>
// CHECK: %[[A:.*]] = linalg.reduce ins(%{{.*}} : tensor<2x3x4x5xf32>) outs(%[[ONES3]] : tensor<2x3x4xf32>) dimensions = [3]
// CHECK: %[[ONES2:.*]] = arith.constant dense<1.000000e+00> : tensor<2x3xf32>
// CHECK: %[[B:.*]] = linalg.reduce ins(%[[A]] : tensor<2x3x4xf32>) outs(%[[ONES2]] : tensor<2x3xf32>) dimensions = [2]
// CHECK: linalg.reduce ins(%[[B]] : tensor<2x3xf32>) outs(%[[INIT]] : tensor<3xf32>) dimensions = [0]
func.func @product(%arg0: tensor<2x3x4x5xf32> {secret.secret}, %init: tensor<3xf32>) -> tensor<3xf32> {
  %0 = linalg.reduce ins(%arg0 : tensor<2x3x4x5xf32>) outs(%init : tensor<3xf32>) dimensions = [0, 2, 3]
    (%in: f32, %acc: f32) {
      %1 = arith.mulf %in, %acc : f32
      linalg.yield %1 : f32
    }
  return %0 : tensor<3xf32>
}
