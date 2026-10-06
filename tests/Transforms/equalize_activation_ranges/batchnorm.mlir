// RUN: heir-opt --equalize-activation-ranges %s | FileCheck %s

#map = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#chan = affine_map<(d0, d1, d2, d3) -> (d1, 0, 0)>

// An unfused BatchNorm2d between a convolution and a ReLU, as torch-mlir
// exports it. 1/sqrt(var + eps) is computed by public generics, one of which
// contains a cf.assert. The ReLU domain [-4, 4] gives factor 8: the conv
// weights, the mean and beta are divided by 8, invstd and gamma are
// multipliers and stay, and the final matmul weights are multiplied by 8.

// CHECK: func.func @batchnorm
// CHECK-DAG: arith.constant dense<5.000000e-01> : tensor<2x1x1x1xf32>
// CHECK-DAG: arith.constant dense<1.250000e-01> : tensor<2xf32>
// CHECK-DAG: arith.constant dense<4.000000e+00> : tensor<2xf32>
// CHECK-DAG: arith.constant dense<2.000000e+00> : tensor<2xf32>
// CHECK-DAG: arith.constant dense<3.750000e-01> : tensor<2xf32>
// CHECK-DAG: arith.constant dense<4.000000e+00> : tensor<8x1xf32>
// CHECK: cf.assert
// CHECK: domain_lower = -5.000000e-01 : f64, domain_upper = 5.000000e-01 : f64
// CHECK: linalg.matmul
func.func @batchnorm(%arg0: tensor<1x1x2x2xf32> {secret.secret}) -> tensor<1x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %one = arith.constant 1.000000e+00 : f32
  %eps = arith.constant 1.000000e-05 : f64
  %w = arith.constant dense<4.0> : tensor<2x1x1x1xf32>
  %mean = arith.constant dense<1.0> : tensor<2xf32>
  %var = arith.constant dense<4.0> : tensor<2xf32>
  %gamma = arith.constant dense<2.0> : tensor<2xf32>
  %beta = arith.constant dense<3.0> : tensor<2xf32>
  %w_fc = arith.constant dense<0.5> : tensor<8x1xf32>
  %empty = tensor.empty() : tensor<1x2x2x2xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %conv = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %w : tensor<1x1x2x2xf32>, tensor<2x1x1x1xf32>) outs(%fill : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %empty_c = tensor.empty() : tensor<2xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%var : tensor<2xf32>) outs(%empty_c : tensor<2xf32>) {
  ^bb0(%in: f32, %out: f32):
    %e = arith.truncf %eps : f64 to f32
    %s = arith.addf %in, %e : f32
    linalg.yield %s : f32
  } -> tensor<2xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%0 : tensor<2xf32>) outs(%empty_c : tensor<2xf32>) {
  ^bb0(%in: f32, %out: f32):
    %s = math.sqrt %in : f32
    linalg.yield %s : f32
  } -> tensor<2xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%1 : tensor<2xf32>) outs(%empty_c : tensor<2xf32>) {
  ^bb0(%in: f32, %out: f32):
    %nonzero = arith.cmpf one, %in, %cst : f32
    cf.assert %nonzero, "unimplemented: tensor with zero element"
    %r = arith.divf %one, %in : f32
    linalg.yield %r : f32
  } -> tensor<2xf32>
  %mean_e = tensor.expand_shape %mean [[0, 1, 2]] output_shape [2, 1, 1] : tensor<2xf32> into tensor<2x1x1xf32>
  %invstd_e = tensor.expand_shape %2 [[0, 1, 2]] output_shape [2, 1, 1] : tensor<2xf32> into tensor<2x1x1xf32>
  %gamma_e = tensor.expand_shape %gamma [[0, 1, 2]] output_shape [2, 1, 1] : tensor<2xf32> into tensor<2x1x1xf32>
  %beta_e = tensor.expand_shape %beta [[0, 1, 2]] output_shape [2, 1, 1] : tensor<2xf32> into tensor<2x1x1xf32>
  %3 = linalg.generic {indexing_maps = [#map4, #chan, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv, %mean_e : tensor<1x2x2x2xf32>, tensor<2x1x1xf32>) outs(%empty : tensor<1x2x2x2xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %s = arith.subf %in, %in_0 : f32
    linalg.yield %s : f32
  } -> tensor<1x2x2x2xf32>
  %4 = linalg.generic {indexing_maps = [#map4, #chan, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3, %invstd_e : tensor<1x2x2x2xf32>, tensor<2x1x1xf32>) outs(%empty : tensor<1x2x2x2xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %s = arith.mulf %in, %in_0 : f32
    linalg.yield %s : f32
  } -> tensor<1x2x2x2xf32>
  %5 = linalg.generic {indexing_maps = [#map4, #chan, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %gamma_e : tensor<1x2x2x2xf32>, tensor<2x1x1xf32>) outs(%empty : tensor<1x2x2x2xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %s = arith.mulf %in, %in_0 : f32
    linalg.yield %s : f32
  } -> tensor<1x2x2x2xf32>
  %6 = linalg.generic {indexing_maps = [#map4, #chan, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5, %beta_e : tensor<1x2x2x2xf32>, tensor<2x1x1xf32>) outs(%empty : tensor<1x2x2x2xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %s = arith.addf %in, %in_0 : f32
    linalg.yield %s : f32
  } -> tensor<1x2x2x2xf32>
  %7 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<1x2x2x2xf32>) outs(%empty : tensor<1x2x2x2xf32>) attrs = {domain_lower = -4.0 : f64, domain_upper = 4.0 : f64} {
  ^bb0(%in: f32, %out: f32):
    %c = arith.cmpf ugt, %in, %cst : f32
    %s = arith.select %c, %in, %cst : f32
    linalg.yield %s : f32
  } -> tensor<1x2x2x2xf32>
  %collapsed = tensor.collapse_shape %7 [[0], [1, 2, 3]] : tensor<1x2x2x2xf32> into tensor<1x8xf32>
  %empty_o = tensor.empty() : tensor<1x1xf32>
  %fill_o = linalg.fill ins(%cst : f32) outs(%empty_o : tensor<1x1xf32>) -> tensor<1x1xf32>
  %8 = linalg.matmul ins(%collapsed, %w_fc : tensor<1x8xf32>, tensor<8x1xf32>) outs(%fill_o : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %8 : tensor<1x1xf32>
}
