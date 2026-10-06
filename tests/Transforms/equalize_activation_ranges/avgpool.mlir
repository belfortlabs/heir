// RUN: heir-opt --equalize-activation-ranges %s | FileCheck %s

#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// An average pool as torch-mlir exports it: a sum pool into a zero fill,
// then a division by the window size. Both commute with a scale, so the
// factor 8 of the ReLU domain [-4, 4] reaches the matmul, whose weights are
// multiplied by 8. The conv weights are divided by 8, and the checkpoint on
// the ReLU output records the factor.

// CHECK: func.func @avgpool
// CHECK-DAG: arith.constant dense<2.500000e-01> : tensor<1x1x1x1xf32>
// CHECK-DAG: arith.constant dense<4.000000e+00> : tensor<1x1xf32>
// CHECK: domain_lower = -5.000000e-01 : f64, domain_upper = 5.000000e-01 : f64
// CHECK: debug.validate {{.*}}debug.scale = 8.000000e+00 : f64
// CHECK: linalg.pooling_nchw_sum
// CHECK: linalg.matmul
func.func @avgpool(%arg0: tensor<1x1x2x2xf32> {secret.secret}) -> tensor<1x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %four = arith.constant 4.000000e+00 : f32
  %w = arith.constant dense<2.0> : tensor<1x1x1x1xf32>
  %w_fc = arith.constant dense<0.5> : tensor<1x1xf32>
  %empty = tensor.empty() : tensor<1x1x2x2xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
  %conv = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %w : tensor<1x1x2x2xf32>, tensor<1x1x1x1xf32>) outs(%fill : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
  %relu = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv : tensor<1x1x2x2xf32>) outs(%empty : tensor<1x1x2x2xf32>) attrs = {domain_lower = -4.0 : f64, domain_upper = 4.0 : f64} {
  ^bb0(%in: f32, %out: f32):
    %c = arith.cmpf ugt, %in, %cst : f32
    %s = arith.select %c, %in, %cst : f32
    linalg.yield %s : f32
  } -> tensor<1x1x2x2xf32>
  debug.validate %relu {name = "relu"} : tensor<1x1x2x2xf32>
  %window = tensor.empty() : tensor<2x2xf32>
  %empty_p = tensor.empty() : tensor<1x1x1x1xf32>
  %fill_p = linalg.fill ins(%cst : f32) outs(%empty_p : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
  %pool = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%relu, %window : tensor<1x1x2x2xf32>, tensor<2x2xf32>) outs(%fill_p : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
  %avg = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%pool : tensor<1x1x1x1xf32>) outs(%empty_p : tensor<1x1x1x1xf32>) {
  ^bb0(%in: f32, %out: f32):
    %d = arith.divf %in, %four : f32
    linalg.yield %d : f32
  } -> tensor<1x1x1x1xf32>
  %collapsed = tensor.collapse_shape %avg [[0], [1, 2, 3]] : tensor<1x1x1x1xf32> into tensor<1x1xf32>
  %empty_o = tensor.empty() : tensor<1x1xf32>
  %fill_o = linalg.fill ins(%cst : f32) outs(%empty_o : tensor<1x1xf32>) -> tensor<1x1xf32>
  %mm = linalg.matmul ins(%collapsed, %w_fc : tensor<1x1xf32>, tensor<1x1xf32>) outs(%fill_o : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %mm : tensor<1x1xf32>
}
