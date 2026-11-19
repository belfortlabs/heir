// REQUIRES: cyclops-planner
// RUN: heir-opt --early-bootstrap-placement=level-budget=3 %s | FileCheck %s
// RUN: heir-opt --early-bootstrap-placement=level-budget=4 %s | FileCheck %s --check-prefix=FITS

// After layout propagation, linalg.pooling_ncw_max costs the levels of the
// kernel.max_pool it lowers to. The slot count comes from the module, not from
// the compact result layout, which ends at slot 15: the pool is a window of 2
// over 32 of 64 slots, which drops 2 levels. A budget of 3 levels is not
// enough for the multiplication and the pool, and a budget of 4 levels is.

// CHECK: mgmt.bootstrap
// CHECK: arith.mulf
// CHECK: linalg.pooling_ncw_max

// FITS-NOT: mgmt.bootstrap

#compact = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and slot = 4i1 + i2 and 0 <= i1 <= 3 and 0 <= i2 <= 3 }">
#rowMajor = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and (-8i1 - i2 + slot) mod 32 = 0 and 0 <= i1 <= 3 and 0 <= i2 <= 7 and 0 <= slot <= 63 }">
module attributes {scheme.ckks, scheme.layout_slot_count = 64 : i64} {
  func.func @pool(%arg0: !secret.secret<tensor<1x4x8xf32>> {tensor_ext.layout = #rowMajor}) -> (!secret.secret<tensor<1x4x4xf32>> {tensor_ext.layout = #compact}) {
    %0 = secret.generic(%arg0 : !secret.secret<tensor<1x4x8xf32>>) {
    ^body(%x: tensor<1x4x8xf32>):
      %sq = arith.mulf %x, %x {tensor_ext.layout = #rowMajor} : tensor<1x4x8xf32>
      %neg_inf = arith.constant 0xFF800000 : f32
      %empty = tensor.empty() : tensor<1x4x4xf32>
      %init = linalg.fill ins(%neg_inf : f32) outs(%empty : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
      %window = tensor.empty() : tensor<2xf32>
      %pool = linalg.pooling_ncw_max {dilations = dense<1> : vector<1xi64>, domain_lower = -1.0 : f64, domain_upper = 1.0 : f64, strides = dense<2> : vector<1xi64>, tensor_ext.layout = #compact} ins(%sq, %window : tensor<1x4x8xf32>, tensor<2xf32>) outs(%init : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
      secret.yield %pool : tensor<1x4x4xf32>
    } -> !secret.secret<tensor<1x4x4xf32>>
    return %0 : !secret.secret<tensor<1x4x4xf32>>
  }
}
