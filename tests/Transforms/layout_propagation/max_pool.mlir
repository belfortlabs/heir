// RUN: heir-opt --layout-propagation=min-slot-count=64 --split-input-file %s | FileCheck %s

// The input is row major and may repeat after the data; the pool keeps it.
// The output is compact: 16 values in the first 16 slots, with no copies.

// CHECK-DAG: [[compact:#.*]] = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and slot = 4i1 + i2 and 0 <= i1 <= 3 and 0 <= i2 <= 3 }">
// CHECK-DAG: [[rowMajor:#.*]] = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and (-8i1 - i2 + slot) mod 32 = 0 and 0 <= i1 <= 3 and 0 <= i2 <= 7 and 0 <= slot <= 63 }">
// CHECK: func.func @pool
// CHECK-SAME: tensor_ext.layout = [[rowMajor]]
// CHECK-NOT: tensor_ext.convert_layout
// CHECK: linalg.pooling_ncw_max
// CHECK-SAME: heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 1, 4, 4>}
// CHECK-SAME: tensor_ext.layout = [[compact]]
module attributes {backend.cheddar} {
  func.func @pool(%arg0: !secret.secret<tensor<1x4x8xf32>>) -> !secret.secret<tensor<1x4x4xf32>> {
    %neg_inf = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<1x4x4xf32>
    %1 = linalg.fill ins(%neg_inf : f32) outs(%0 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    %window = tensor.empty() : tensor<2xf32>
    %r = secret.generic(%arg0 : !secret.secret<tensor<1x4x8xf32>>) {
    ^body(%x: tensor<1x4x8xf32>):
      %2 = linalg.pooling_ncw_max {dilations = dense<1> : vector<1xi64>, domain_lower = 0.0 : f64, domain_upper = 1.0 : f64, strides = dense<2> : vector<1xi64>} ins(%x, %window : tensor<1x4x8xf32>, tensor<2xf32>) outs(%1 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
      secret.yield %2 : tensor<1x4x4xf32>
    } -> !secret.secret<tensor<1x4x4xf32>>
    return %r : !secret.secret<tensor<1x4x4xf32>>
  }
}

// -----

// A compact input, such as the output of another max pool, needs no
// conversion either.

// CHECK: func.func @chained
// CHECK-NOT: tensor_ext.convert_layout
// CHECK: linalg.pooling_ncw_max
// CHECK-NOT: tensor_ext.convert_layout
// CHECK: linalg.pooling_ncw_max
module attributes {backend.cheddar} {
  func.func @chained(%arg0: !secret.secret<tensor<1x4x8xf32>>) -> !secret.secret<tensor<1x4x2xf32>> {
    %neg_inf = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<1x4x4xf32>
    %1 = linalg.fill ins(%neg_inf : f32) outs(%0 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    %2 = tensor.empty() : tensor<1x4x2xf32>
    %3 = linalg.fill ins(%neg_inf : f32) outs(%2 : tensor<1x4x2xf32>) -> tensor<1x4x2xf32>
    %window = tensor.empty() : tensor<2xf32>
    %r = secret.generic(%arg0 : !secret.secret<tensor<1x4x8xf32>>) {
    ^body(%x: tensor<1x4x8xf32>):
      %4 = linalg.pooling_ncw_max {dilations = dense<1> : vector<1xi64>, domain_lower = 0.0 : f64, domain_upper = 1.0 : f64, strides = dense<2> : vector<1xi64>} ins(%x, %window : tensor<1x4x8xf32>, tensor<2xf32>) outs(%1 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
      %5 = linalg.pooling_ncw_max {dilations = dense<1> : vector<1xi64>, domain_lower = 0.0 : f64, domain_upper = 1.0 : f64, strides = dense<2> : vector<1xi64>} ins(%4, %window : tensor<1x4x4xf32>, tensor<2xf32>) outs(%3 : tensor<1x4x2xf32>) -> tensor<1x4x2xf32>
      secret.yield %5 : tensor<1x4x2xf32>
    } -> !secret.secret<tensor<1x4x2xf32>>
    return %r : !secret.secret<tensor<1x4x2xf32>>
  }
}

// -----

// Any other input layout is converted to row major first.

// CHECK: func.func @after_transpose
// CHECK: linalg.transpose
// CHECK: tensor_ext.convert_layout
// CHECK: linalg.pooling_ncw_max
module attributes {backend.cheddar} {
  func.func @after_transpose(%arg0: !secret.secret<tensor<1x8x4xf32>>) -> !secret.secret<tensor<1x4x4xf32>> {
    %neg_inf = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<1x4x4xf32>
    %1 = linalg.fill ins(%neg_inf : f32) outs(%0 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    %t = tensor.empty() : tensor<1x4x8xf32>
    %window = tensor.empty() : tensor<2xf32>
    %r = secret.generic(%arg0 : !secret.secret<tensor<1x8x4xf32>>) {
    ^body(%x: tensor<1x8x4xf32>):
      %2 = linalg.transpose ins(%x : tensor<1x8x4xf32>) outs(%t : tensor<1x4x8xf32>) permutation = [0, 2, 1]
      %3 = linalg.pooling_ncw_max {dilations = dense<1> : vector<1xi64>, domain_lower = 0.0 : f64, domain_upper = 1.0 : f64, strides = dense<2> : vector<1xi64>} ins(%2, %window : tensor<1x4x8xf32>, tensor<2xf32>) outs(%1 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
      secret.yield %3 : tensor<1x4x4xf32>
    } -> !secret.secret<tensor<1x4x4xf32>>
    return %r : !secret.secret<tensor<1x4x4xf32>>
  }
}
