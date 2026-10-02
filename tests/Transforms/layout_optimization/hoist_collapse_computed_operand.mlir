// RUN: heir-opt --layout-optimization --canonicalize %s | FileCheck %s

// Hoisting a convert_layout above a collapse_shape whose operand is computed
// (by an op layout-optimization cannot hoist through, so its layout is not free) costs converting that operand to the hoisted,
// rank-3 layout. Costing it against the rank-2 result layout instead composed
// relations of different ranks and crashed.

#before3 = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and slot = 8i1 + i2 and 0 <= i1 <= 3 and 0 <= i2 <= 7 }">
#before2 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and slot = 8i0 + i1 and 0 <= i0 <= 3 and 0 <= i1 <= 7 }">
#target2 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and slot = i0 + 4i1 and 0 <= i0 <= 3 and 0 <= i1 <= 7 }">
module {
  // CHECK: func.func @collapse_hoist_computed
  // CHECK: %[[div:.*]] = arith.divf
  // CHECK-NEXT: %[[conv:.*]] = tensor_ext.convert_layout %[[div]]
  // CHECK-NEXT: tensor.collapse_shape %[[conv]]
  // CHECK-NEXT: secret.yield
  func.func @collapse_hoist_computed(%arg0: !secret.secret<tensor<1x4x8xf32>> {tensor_ext.layout = #before3}) -> (!secret.secret<tensor<4x8xf32>> {tensor_ext.layout = #target2}) {
    %0 = secret.generic(%arg0: !secret.secret<tensor<1x4x8xf32>> {tensor_ext.layout = #before3}) {
    ^body(%input0: tensor<1x4x8xf32>):
      %sum = arith.divf %input0, %input0 {tensor_ext.layout = #before3} : tensor<1x4x8xf32>
      %collapsed = tensor.collapse_shape %sum [[0, 1], [2]] {tensor_ext.layout = #before2} : tensor<1x4x8xf32> into tensor<4x8xf32>
      %1 = tensor_ext.convert_layout %collapsed {from_layout = #before2, tensor_ext.layout = #target2, to_layout = #target2} : tensor<4x8xf32>
      secret.yield %1 : tensor<4x8xf32>
    } -> (!secret.secret<tensor<4x8xf32>> {tensor_ext.layout = #target2})
    return %0 : !secret.secret<tensor<4x8xf32>>
  }
}
