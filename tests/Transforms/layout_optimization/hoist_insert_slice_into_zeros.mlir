// RUN: heir-opt --layout-optimization=min-slot-count=16 %s | FileCheck %s

// A conversion after a partial insert into zeros hoists through the insert: the
// source takes the target layout seen through the insertion, and the zeros
// need no layout. Here the source is a function argument, so the client packs
// it in that layout and the conversion disappears.

#src = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 4 = 0 and 0 <= i0 <= 2 and 0 <= slot <= 15 }">
#inserted = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (1 + i0) mod 2 = 0 and (1 - i0 + 2slot) mod 8 = 0 and 0 < i0 <= 5 and 0 <= slot <= 15 }">
#row_major = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 8 = 0 and 0 <= i0 <= 6 and 0 <= slot <= 15 }">

// CHECK: #[[row_major:.*]] = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 8 = 0 and 0 <= i0 <= 6 and 0 <= slot <= 15 }">
// CHECK: #[[hoisted:.*]] = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-1 - 2i0 + slot) mod 8 = 0 and 0 <= i0 <= 2 and 0 <= slot <= 15 }">

module {
  // CHECK: func.func @stride_2
  // CHECK-SAME: tensor_ext.layout = #[[hoisted]]
  // CHECK-NOT: tensor_ext.assign_layout
  // CHECK: tensor.insert_slice
  // CHECK-SAME: tensor_ext.layout = #[[row_major]]
  // CHECK-NOT: tensor_ext.convert_layout
  // CHECK: arith.addf
  func.func @stride_2(%arg0: !secret.secret<tensor<3xf32>> {tensor_ext.layout = #src}) -> (!secret.secret<tensor<7xf32>> {tensor_ext.layout = #row_major}) {
    %zeros = arith.constant dense<0.000000e+00> : tensor<7xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<3xf32>> {tensor_ext.layout = #src}) {
    ^body(%input0: tensor<3xf32>):
      %1 = tensor.insert_slice %input0 into %zeros[1] [3] [2] {tensor_ext.layout = #inserted} : tensor<3xf32> into tensor<7xf32>
      %2 = tensor_ext.convert_layout %1 {from_layout = #inserted, tensor_ext.layout = #row_major, to_layout = #row_major} : tensor<7xf32>
      %3 = arith.addf %2, %2 {tensor_ext.layout = #row_major} : tensor<7xf32>
      secret.yield %3 : tensor<7xf32>
    } -> (!secret.secret<tensor<7xf32>> {tensor_ext.layout = #row_major})
    return %0 : !secret.secret<tensor<7xf32>>
  }
}
