// RUN: heir-opt --split-input-file --activation-canonicalizations %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>

// ReLU6 as torch-mlir exports it: a maximum and a minimum generic with the
// bound as a 0-d tensor, compared with ordered predicates. Each half becomes
// a maximumf/minimumf carrying its own generic's domain.
// CHECK: func.func @relu6
// CHECK: linalg.generic
// CHECK-NOT: domain_lower
// CHECK: arith.maximumf %{{.*}}, %{{.*}} {domain_lower = -4.000000e+00 : f64, domain_upper = 8.000000e+00 : f64}
// CHECK: linalg.generic
// CHECK-NOT: domain_lower
// CHECK: arith.minimumf %{{.*}}, %{{.*}} {domain_lower = 0.000000e+00 : f64, domain_upper = 8.000000e+00 : f64}
// CHECK-NOT: arith.select
// CHECK: return
func.func @relu6(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> {
  %cst_0 = arith.constant dense<6.000000e+00> : tensor<f32>
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = tensor.empty() : tensor<1x3xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"], domain_lower = -4.0 : f64, domain_upper = 8.0 : f64} ins(%arg0, %cst_1 : tensor<1x3xf32>, tensor<f32>) outs(%0 : tensor<1x3xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.cmpf ogt, %in, %in_0 : f32
    %4 = arith.select %3, %in, %in_0 : f32
    linalg.yield %4 : f32
  } -> tensor<1x3xf32>
  %2 = linalg.generic {indexing_maps = [#map1, #map, #map], iterator_types = ["parallel", "parallel"], domain_lower = 0.0 : f64, domain_upper = 8.0 : f64} ins(%cst_0, %1 : tensor<f32>, tensor<1x3xf32>) outs(%0 : tensor<1x3xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.cmpf olt, %in, %in_0 : f32
    %4 = arith.select %3, %in, %in_0 : f32
    linalg.yield %4 : f32
  } -> tensor<1x3xf32>
  return %2 : tensor<1x3xf32>
}

// -----

// Scalar forms, including non-strict predicates.
// CHECK: func.func @min_max_scalar
// CHECK-SAME: (%[[ARG0:.*]]: f32)
// CHECK: %[[C1:.*]] = arith.constant 1.0
// CHECK: %[[C0:.*]] = arith.constant 0.0
// CHECK: %[[MIN:.*]] = arith.minimumf %[[ARG0]], %[[C1]]
// CHECK: %[[MAX:.*]] = arith.maximumf %[[MIN]], %[[C0]]
// CHECK: return %[[MAX]]
func.func @min_max_scalar(%arg0: f32) -> f32 {
  %c1 = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0.000000e+00 : f32
  %0 = arith.cmpf ole, %arg0, %c1 : f32
  %1 = arith.select %0, %arg0, %c1 : f32
  %2 = arith.cmpf oge, %1, %c0 : f32
  %3 = arith.select %2, %1, %c0 : f32
  return %3 : f32
}

// -----

// torch-mlir's aten.clamp lowering: select(x < lo, lo, x) = max(x, lo) and
// select(x > hi, hi, x) = min(x, hi).
// CHECK: func.func @torch_clamp
// CHECK-NOT: arith.select
// CHECK: %[[MAX:.*]] = arith.maximumf %arg0, %{{.*}} : f32
// CHECK: arith.minimumf %[[MAX]], %{{.*}} : f32
func.func @torch_clamp(%arg0: f32) -> f32 {
  %c0 = arith.constant 0.000000e+00 : f32
  %c6 = arith.constant 6.000000e+00 : f32
  %0 = arith.cmpf ult, %arg0, %c0 : f32
  %1 = arith.select %0, %c0, %arg0 : f32
  %2 = arith.cmpf ugt, %1, %c6 : f32
  %3 = arith.select %2, %c6, %1 : f32
  return %3 : f32
}

// -----

// A max of two non-constant values has no lowering on ciphertexts, so it is
// left to the compare-to-sign path.
// CHECK: func.func @two_variables
// CHECK: arith.cmpf
// CHECK: arith.select
func.func @two_variables(%arg0: f32, %arg1: f32) -> f32 {
  %0 = arith.cmpf ogt, %arg0, %arg1 : f32
  %1 = arith.select %0, %arg0, %arg1 : f32
  return %1 : f32
}
