// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

// Cyclops scales the values into its comparison domain, so the pass does not
// normalize the input of a max pool.

// CHECK: func.func @floor_mode
// CHECK-SAME: (%[[arg0:.*]]: tensor<1x4x8xf32>)
// CHECK-NOT: arith.mulf
// CHECK: linalg.pooling_ncw_max
// CHECK-SAME: domain_lower = -2.000000e+00 : f64
// CHECK-SAME: domain_upper = 6.000000e+00 : f64
// CHECK-SAME: ins(%[[arg0]],
// CHECK-NOT: arith.mulf
// CHECK: return
func.func @floor_mode(%arg0: tensor<1x4x8xf32>) -> tensor<1x4x4xf32> {
  %neg_inf = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<1x4x4xf32>
  %1 = linalg.fill ins(%neg_inf : f32) outs(%0 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  %window = tensor.empty() : tensor<2xf32>
  %2 = linalg.pooling_ncw_max {dilations = dense<1> : vector<1xi64>, domain_lower = -2.0 : f64, domain_upper = 6.0 : f64, strides = dense<2> : vector<1xi64>} ins(%arg0, %window : tensor<1x4x8xf32>, tensor<2xf32>) outs(%1 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  return %2 : tensor<1x4x4xf32>
}

// -----

// A ceil-mode pool reads a -inf pad. With a negative domain_lower, the pad
// becomes a zero pad plus a constant that holds domain_lower in the padded
// positions.

// CHECK: func.func @ceil_mode_negative_domain
// CHECK-SAME: (%[[arg0:.*]]: tensor<1x2x3xf32>)
// CHECK-DAG: %[[zero:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[padding:.*]] = arith.constant dense<{{\[\[\[}}0.000000e+00, 0.000000e+00, 0.000000e+00, -2.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, -2.000000e+00]]]> : tensor<1x2x4xf32>
// CHECK: %[[padded:.*]] = tensor.pad %[[arg0]] low[0, 0, 0] high[0, 0, 1]
// CHECK: tensor.yield %[[zero]]
// CHECK: %[[input:.*]] = arith.addf %[[padded]], %[[padding]]
// CHECK: linalg.pooling_ncw_max
// CHECK-SAME: ins(%[[input]],
func.func @ceil_mode_negative_domain(%arg0: tensor<1x2x3xf32>) -> tensor<1x2x2xf32> {
  %neg_inf = arith.constant 0xFF800000 : f32
  %padded = tensor.pad %arg0 low[0, 0, 0] high[0, 0, 1] {
  ^bb0(%i: index, %j: index, %k: index):
    tensor.yield %neg_inf : f32
  } : tensor<1x2x3xf32> to tensor<1x2x4xf32>
  %0 = tensor.empty() : tensor<1x2x2xf32>
  %1 = linalg.fill ins(%neg_inf : f32) outs(%0 : tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
  %window = tensor.empty() : tensor<2xf32>
  %2 = linalg.pooling_ncw_max {dilations = dense<1> : vector<1xi64>, domain_lower = -2.0 : f64, domain_upper = 6.0 : f64, strides = dense<2> : vector<1xi64>} ins(%padded, %window : tensor<1x2x4xf32>, tensor<2xf32>) outs(%1 : tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
  return %2 : tensor<1x2x2xf32>
}

// -----

// Low padding gets the lower bound too. FoldConstantFill may turn the -inf
// fill into a splat constant first; the pattern accepts that form.

// CHECK: func.func @low_and_high_padding
// CHECK-DAG: %[[padding:.*]] = arith.constant dense<{{\[\[\[}}-5.000000e-01, 0.000000e+00, 0.000000e+00, -5.000000e-01]]]> : tensor<1x1x4xf32>
// CHECK: %[[padded:.*]] = tensor.pad %{{.*}} low[0, 0, 1] high[0, 0, 1]
// CHECK: %[[input:.*]] = arith.addf %[[padded]], %[[padding]]
// CHECK: linalg.pooling_ncw_max
// CHECK-SAME: ins(%[[input]],
func.func @low_and_high_padding(%arg0: tensor<1x1x2xf32>) -> tensor<1x1x2xf32> {
  %neg_inf = arith.constant 0xFF800000 : f32
  %padded = tensor.pad %arg0 low[0, 0, 1] high[0, 0, 1] {
  ^bb0(%i: index, %j: index, %k: index):
    tensor.yield %neg_inf : f32
  } : tensor<1x1x2xf32> to tensor<1x1x4xf32>
  %init = arith.constant dense<0xFF800000> : tensor<1x1x2xf32>
  %window = tensor.empty() : tensor<2xf32>
  %0 = linalg.pooling_ncw_max {dilations = dense<1> : vector<1xi64>, domain_lower = -0.5 : f64, domain_upper = 0.25 : f64, strides = dense<2> : vector<1xi64>} ins(%padded, %window : tensor<1x1x4xf32>, tensor<2xf32>) outs(%init : tensor<1x1x2xf32>) -> tensor<1x1x2xf32>
  return %0 : tensor<1x1x2xf32>
}

// -----

// With a domain_lower of zero or more, a zero pad alone is correct: zero is
// not more than a real element.

// CHECK: func.func @ceil_mode_nonnegative_domain
// CHECK-SAME: (%[[arg0:.*]]: tensor<1x4x7xf32>)
// CHECK-DAG: %[[zero:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[padded:.*]] = tensor.pad %[[arg0]] low[0, 0, 0] high[0, 0, 1]
// CHECK: tensor.yield %[[zero]]
// CHECK-NOT: arith.addf
// CHECK: linalg.pooling_ncw_max
// CHECK-SAME: ins(%[[padded]],
func.func @ceil_mode_nonnegative_domain(%arg0: tensor<1x4x7xf32>) -> tensor<1x4x4xf32> {
  %neg_inf = arith.constant 0xFF800000 : f32
  %padded = tensor.pad %arg0 low[0, 0, 0] high[0, 0, 1] {
  ^bb0(%i: index, %j: index, %k: index):
    tensor.yield %neg_inf : f32
  } : tensor<1x4x7xf32> to tensor<1x4x8xf32>
  %0 = tensor.empty() : tensor<1x4x4xf32>
  %1 = linalg.fill ins(%neg_inf : f32) outs(%0 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  %window = tensor.empty() : tensor<2xf32>
  %2 = linalg.pooling_ncw_max {dilations = dense<1> : vector<1xi64>, domain_lower = 0.5 : f64, domain_upper = 3.0 : f64, strides = dense<2> : vector<1xi64>} ins(%padded, %window : tensor<1x4x8xf32>, tensor<2xf32>) outs(%1 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  return %2 : tensor<1x4x4xf32>
}

// -----

// A -inf pad stays when the domain is unknown, or when the init is not -inf
// (the init takes part in the max).

// CHECK: func.func @ceil_mode_no_domain
// CHECK: %[[negInf:.*]] = arith.constant 0xFF800000 : f32
// CHECK: tensor.pad
// CHECK: tensor.yield %[[negInf]]
func.func @ceil_mode_no_domain(%arg0: tensor<1x4x7xf32>) -> tensor<1x4x4xf32> {
  %neg_inf = arith.constant 0xFF800000 : f32
  %padded = tensor.pad %arg0 low[0, 0, 0] high[0, 0, 1] {
  ^bb0(%i: index, %j: index, %k: index):
    tensor.yield %neg_inf : f32
  } : tensor<1x4x7xf32> to tensor<1x4x8xf32>
  %0 = tensor.empty() : tensor<1x4x4xf32>
  %1 = linalg.fill ins(%neg_inf : f32) outs(%0 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  %window = tensor.empty() : tensor<2xf32>
  %2 = linalg.pooling_ncw_max {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>} ins(%padded, %window : tensor<1x4x8xf32>, tensor<2xf32>) outs(%1 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  return %2 : tensor<1x4x4xf32>
}

// CHECK: func.func @ceil_mode_finite_init
// CHECK: %[[negInf:.*]] = arith.constant 0xFF800000 : f32
// CHECK: tensor.pad
// CHECK: tensor.yield %[[negInf]]
func.func @ceil_mode_finite_init(%arg0: tensor<1x4x7xf32>) -> tensor<1x4x4xf32> {
  %neg_inf = arith.constant 0xFF800000 : f32
  %zero = arith.constant 0.0 : f32
  %padded = tensor.pad %arg0 low[0, 0, 0] high[0, 0, 1] {
  ^bb0(%i: index, %j: index, %k: index):
    tensor.yield %neg_inf : f32
  } : tensor<1x4x7xf32> to tensor<1x4x8xf32>
  %0 = tensor.empty() : tensor<1x4x4xf32>
  %1 = linalg.fill ins(%zero : f32) outs(%0 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  %window = tensor.empty() : tensor<2xf32>
  %2 = linalg.pooling_ncw_max {dilations = dense<1> : vector<1xi64>, domain_lower = -2.0 : f64, domain_upper = 6.0 : f64, strides = dense<2> : vector<1xi64>} ins(%padded, %window : tensor<1x4x8xf32>, tensor<2xf32>) outs(%1 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  return %2 : tensor<1x4x4xf32>
}

// -----

// A floor-mode pool over 75 elements with k = 2 reads only 74 of them, so the
// last one is sliced off.

// CHECK: func.func @floor_mode_tail
// CHECK-SAME: (%[[arg0:.*]]: tensor<1x4x75xf32>)
// CHECK: %[[trimmed:.*]] = tensor.extract_slice %[[arg0]][0, 0, 0] [1, 4, 74] [1, 1, 1]
// CHECK: linalg.pooling_ncw_max
// CHECK-SAME: ins(%[[trimmed]],
func.func @floor_mode_tail(%arg0: tensor<1x4x75xf32>) -> tensor<1x4x37xf32> {
  %neg_inf = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<1x4x37xf32>
  %1 = linalg.fill ins(%neg_inf : f32) outs(%0 : tensor<1x4x37xf32>) -> tensor<1x4x37xf32>
  %window = tensor.empty() : tensor<2xf32>
  %2 = linalg.pooling_ncw_max {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>} ins(%arg0, %window : tensor<1x4x75xf32>, tensor<2xf32>) outs(%1 : tensor<1x4x37xf32>) -> tensor<1x4x37xf32>
  return %2 : tensor<1x4x37xf32>
}
