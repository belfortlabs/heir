// RUN: heir-opt --reduction-canonicalizations --split-input-file %s | FileCheck %s

// A global average pool's sum over H and W becomes a collapse of H and W and
// one reduction over the collapsed dimension, with the original init.

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0)>
// CHECK: func @spatial_sum
// CHECK: %[[INIT:.*]] = arith.constant dense<1.000000e+00> : tensor<72xf32>
// CHECK: %[[FLAT:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0], [1, 2]] : tensor<72x4x4xf32> into tensor<72x16xf32>
// CHECK: linalg.reduce ins(%[[FLAT]] : tensor<72x16xf32>) outs(%[[INIT]] : tensor<72xf32>) dimensions = [1]
// CHECK: arith.addf
// CHECK-NOT: linalg.reduce
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

// An integer product over the last three dimensions.

// CHECK: func @int_product
// CHECK: tensor.collapse_shape %{{.*}} {{\[\[}}0], [1, 2, 3]] : tensor<2x3x4x5xi32> into tensor<2x60xi32>
// CHECK: linalg.reduce { arith.muli {{.*}} ins(%{{.*}} : tensor<2x60xi32>) outs(%{{.*}} : tensor<2xi32>) dimensions = [1]
func.func @int_product(%arg0: tensor<2x3x4x5xi32> {secret.secret}, %init: tensor<2xi32>) -> tensor<2xi32> {
  %0 = linalg.reduce ins(%arg0 : tensor<2x3x4x5xi32>) outs(%init : tensor<2xi32>) dimensions = [1, 2, 3]
    (%in: i32, %acc: i32) {
      %1 = arith.muli %acc, %in : i32
      linalg.yield %1 : i32
    }
  return %0 : tensor<2xi32>
}

// -----

// Dimensions that are not a trailing run stay as they are; the
// ciphertext-semantics kernel only reduces the innermost dimension.

// CHECK: func @leading_dims
// CHECK-NOT: tensor.collapse_shape
// CHECK: linalg.reduce ins(%{{.*}} : tensor<2x3x4xf32>) outs(%{{.*}} : tensor<3xf32>) dimensions = [0, 2]
func.func @leading_dims(%arg0: tensor<2x3x4xf32> {secret.secret}, %init: tensor<3xf32>) -> tensor<3xf32> {
  %0 = linalg.reduce ins(%arg0 : tensor<2x3x4xf32>) outs(%init : tensor<3xf32>) dimensions = [0, 2]
    (%in: f32, %acc: f32) {
      %1 = arith.addf %in, %acc : f32
      linalg.yield %1 : f32
    }
  return %0 : tensor<3xf32>
}

// -----

// A body that does not combine the two block arguments stays as it is.

// CHECK: func @not_a_sum
// CHECK-NOT: tensor.collapse_shape
// CHECK: dimensions = [1, 2]
func.func @not_a_sum(%arg0: tensor<2x3x4xf32> {secret.secret}, %init: tensor<2xf32>) -> tensor<2xf32> {
  %0 = linalg.reduce ins(%arg0 : tensor<2x3x4xf32>) outs(%init : tensor<2xf32>) dimensions = [1, 2]
    (%in: f32, %acc: f32) {
      %1 = arith.addf %in, %in : f32
      linalg.yield %1 : f32
    }
  return %0 : tensor<2xf32>
}
