// RUN: heir-opt --split-input-file --equalize-activation-ranges %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#bias = affine_map<(d0, d1) -> (d1)>

// The ReLU domain [-4, 4] is the calibrated range [-2, 2] widened by the
// default margin 2, so its inputs are divided by 2 * 2 / 0.5 = 8. The first
// layer's weights and bias absorb 1/8 and the second layer's weights absorb 8.
// The returned value keeps its scale, so the second bias is not changed.

// CHECK: func.func @mlp
// CHECK-DAG: %[[W0:.*]] = arith.constant dense<{{\[\[}}1.250000e-01, 2.500000e-01], [3.750000e-01, 5.000000e-01]]> : tensor<2x2xf32>
// CHECK-DAG: %[[B0:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
// CHECK-DAG: %[[W1:.*]] = arith.constant dense<{{\[\[}}8.000000e+00, 4.000000e+00], [2.000000e+00, 1.600000e+01]]> : tensor<2x2xf32>
// CHECK-DAG: %[[B1:.*]] = arith.constant dense<[3.000000e+00, 5.000000e+00]> : tensor<2xf32>
// CHECK: linalg.transpose ins(%[[W0]]
// CHECK: linalg.generic {{.*}} ins(%{{.*}}, %[[B0]] :
// CHECK: domain_lower = -5.000000e-01 : f64, domain_upper = 5.000000e-01 : f64
// CHECK: linalg.transpose ins(%[[W1]]
// CHECK: linalg.generic {{.*}} ins(%{{.*}}, %[[B1]] :
func.func @mlp(%arg0: tensor<1x2xf32> {secret.secret}) -> tensor<1x2xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %w0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %b0 = arith.constant dense<[8.0, 16.0]> : tensor<2xf32>
  %w1 = arith.constant dense<[[1.0, 0.5], [0.25, 2.0]]> : tensor<2x2xf32>
  %b1 = arith.constant dense<[3.0, 5.0]> : tensor<2xf32>
  %empty = tensor.empty() : tensor<1x2xf32>
  %empty_w = tensor.empty() : tensor<2x2xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x2xf32>) -> tensor<1x2xf32>
  %w0t = linalg.transpose ins(%w0 : tensor<2x2xf32>) outs(%empty_w : tensor<2x2xf32>) permutation = [1, 0]
  %0 = linalg.matmul ins(%arg0, %w0t : tensor<1x2xf32>, tensor<2x2xf32>) outs(%fill : tensor<1x2xf32>) -> tensor<1x2xf32>
  %1 = linalg.generic {indexing_maps = [#map, #bias, #map], iterator_types = ["parallel", "parallel"]} ins(%0, %b0 : tensor<1x2xf32>, tensor<2xf32>) outs(%empty : tensor<1x2xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %6 = arith.addf %in, %in_0 : f32
    linalg.yield %6 : f32
  } -> tensor<1x2xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<1x2xf32>) outs(%empty : tensor<1x2xf32>) attrs = {domain_lower = -4.0 : f64, domain_upper = 4.0 : f64} {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.cmpf ugt, %in, %cst : f32
    %7 = arith.select %6, %in, %cst : f32
    linalg.yield %7 : f32
  } -> tensor<1x2xf32>
  %w1t = linalg.transpose ins(%w1 : tensor<2x2xf32>) outs(%empty_w : tensor<2x2xf32>) permutation = [1, 0]
  %3 = linalg.matmul ins(%2, %w1t : tensor<1x2xf32>, tensor<2x2xf32>) outs(%fill : tensor<1x2xf32>) -> tensor<1x2xf32>
  %4 = linalg.generic {indexing_maps = [#map, #bias, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %b1 : tensor<1x2xf32>, tensor<2xf32>) outs(%empty : tensor<1x2xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %6 = arith.addf %in, %in_0 : f32
    linalg.yield %6 : f32
  } -> tensor<1x2xf32>
  return %4 : tensor<1x2xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Without domain attributes the program is not changed.

// CHECK: func.func @no_domains
// CHECK-DAG: arith.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>
// CHECK-DAG: arith.constant dense<{{\[\[}}1.000000e+00, 5.000000e-01], [2.500000e-01, 2.000000e+00]]> : tensor<2x2xf32>
func.func @no_domains(%arg0: tensor<1x2xf32> {secret.secret}) -> tensor<1x2xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %w0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %w1 = arith.constant dense<[[1.0, 0.5], [0.25, 2.0]]> : tensor<2x2xf32>
  %empty = tensor.empty() : tensor<1x2xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x2xf32>) -> tensor<1x2xf32>
  %0 = linalg.matmul ins(%arg0, %w0 : tensor<1x2xf32>, tensor<2x2xf32>) outs(%fill : tensor<1x2xf32>) -> tensor<1x2xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<1x2xf32>) outs(%empty : tensor<1x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    %4 = arith.cmpf ugt, %in, %cst : f32
    %5 = arith.select %4, %in, %cst : f32
    linalg.yield %5 : f32
  } -> tensor<1x2xf32>
  %2 = linalg.matmul ins(%1, %w1 : tensor<1x2xf32>, tensor<2x2xf32>) outs(%fill : tensor<1x2xf32>) -> tensor<1x2xf32>
  return %2 : tensor<1x2xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// tanh does not commute with a scale, so its input class keeps factor 1.

// CHECK: func.func @unknown_op
// CHECK: arith.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>
// CHECK: domain_lower = -4.000000e+00 : f64, domain_upper = 4.000000e+00 : f64
func.func @unknown_op(%arg0: tensor<1x2xf32> {secret.secret}) -> tensor<1x2xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %w0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %empty = tensor.empty() : tensor<1x2xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x2xf32>) -> tensor<1x2xf32>
  %0 = linalg.matmul ins(%arg0, %w0 : tensor<1x2xf32>, tensor<2x2xf32>) outs(%fill : tensor<1x2xf32>) -> tensor<1x2xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<1x2xf32>) outs(%empty : tensor<1x2xf32>) attrs = {domain_lower = -4.0 : f64, domain_upper = 4.0 : f64} {
  ^bb0(%in: f32, %out: f32):
    %4 = arith.cmpf ugt, %in, %cst : f32
    %5 = arith.select %4, %in, %cst : f32
    linalg.yield %5 : f32
  } -> tensor<1x2xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<1x2xf32>) outs(%empty : tensor<1x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    %4 = math.tanh %in : f32
    linalg.yield %4 : f32
  } -> tensor<1x2xf32>
  return %2 : tensor<1x2xf32>
}
