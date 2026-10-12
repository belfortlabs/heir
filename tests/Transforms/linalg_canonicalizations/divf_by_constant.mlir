// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Hard-swish divides by a scalar constant defined outside the generic body.
// CHECK: func.func @divf_in_generic
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x2x3x3xf32>)
// CHECK: %[[INV:.*]] = arith.constant dense<0.166666672> : tensor<1x2x3x3xf32>
// CHECK-NOT: arith.divf
// CHECK: %[[RES:.*]] = arith.mulf %[[ARG0]], %[[INV]]
// CHECK: return %[[RES]]
func.func @divf_in_generic(%arg0: tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32> {
  %cst = arith.constant 6.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x2x3x3xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x2x3x3xf32>) outs(%0 : tensor<1x2x3x3xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.divf %in, %cst : f32
    linalg.yield %2 : f32
  } -> tensor<1x2x3x3xf32>
  return %1 : tensor<1x2x3x3xf32>
}

// -----

// CHECK: func.func @divf_scalar
// CHECK-SAME: (%[[ARG0:.*]]: f32)
// CHECK: %[[INV:.*]] = arith.constant 1.250000e-01 : f32
// CHECK: %[[RES:.*]] = arith.mulf %[[ARG0]], %[[INV]] {some_attr = "foo"} : f32
// CHECK: return %[[RES]]
func.func @divf_scalar(%arg0: f32) -> f32 {
  %cst = arith.constant 8.000000e+00 : f32
  %0 = arith.divf %arg0, %cst {some_attr = "foo"} : f32
  return %0 : f32
}

// -----

// A non-splat divisor is inverted elementwise.
// CHECK: func.func @divf_dense
// CHECK-SAME: (%[[ARG0:.*]]: tensor<3xf32>)
// CHECK: %[[INV:.*]] = arith.constant dense<[5.000000e-01, 2.500000e-01, 2.000000e+00]> : tensor<3xf32>
// CHECK: %[[RES:.*]] = arith.mulf %[[ARG0]], %[[INV]]
// CHECK: return %[[RES]]
func.func @divf_dense(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst = arith.constant dense<[2.0, 4.0, 0.5]> : tensor<3xf32>
  %0 = arith.divf %arg0, %cst : tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

// Division by zero and by a non-constant stay divisions.
// CHECK: func.func @divf_not_rewritten
// CHECK: arith.divf
// CHECK: arith.divf
// CHECK-NOT: arith.mulf
func.func @divf_not_rewritten(%arg0: f32, %arg1: f32) -> f32 {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = arith.divf %arg0, %cst : f32
  %1 = arith.divf %0, %arg1 : f32
  return %1 : f32
}
