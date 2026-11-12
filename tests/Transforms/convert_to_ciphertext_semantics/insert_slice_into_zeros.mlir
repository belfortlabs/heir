// RUN: heir-opt --convert-to-ciphertext-semantics=min-slot-count=16 %s | FileCheck %s

// A partial insert into zeros keeps the source ciphertext: it lowers to a
// remap from the source layout to the result layout, with no mask
// multiplication. The zeros are a linalg.fill, as torch-mlir emits them.

// CHECK: func.func @fill_stride_2_1
// CHECK-NOT: arith.mulf
// CHECK: tensor_ext.remap
// CHECK-NOT: arith.mulf
// CHECK-NOT: arith.addf
// CHECK: secret.yield

#layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (1 + i0) mod 2 = 0 and (7 - 3i0 - 2i1 + 2slot) mod 16 = 0 and 0 < i0 <= 3 and 2 <= i1 <= 4 and 0 <= slot <= 15 and 32*floor((10 + 3i0 + 2i1)/32) <= -7 + 3i0 + 2i1 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-3i0 - i1 + slot) mod 8 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 15 - 3i0 and i1 <= 2 and 0 <= slot <= 15 and 16*floor((8 + 3i0 + i1)/16) <= 3i0 + i1 }">
module attributes {scheme.layout_slot_count = 16 : i64} {
  func.func @fill_stride_2_1(%arg0: !secret.secret<tensor<2x3xf32>> {heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 2, 3>}, tensor_ext.layout = #layout1}) -> (!secret.secret<tensor<4x5xf32>> {tensor_ext.layout = #layout}) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x5xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x5xf32>) -> tensor<4x5xf32>
    %2 = secret.generic(%arg0: !secret.secret<tensor<2x3xf32>> {heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 2, 3>}, tensor_ext.layout = #layout1}) {
    ^body(%input0: tensor<2x3xf32>):
      %inserted_slice = tensor.insert_slice %input0 into %1[1, 2] [2, 3] [2, 1] {heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 4, 5>}, tensor_ext.layout = #layout} : tensor<2x3xf32> into tensor<4x5xf32>
      secret.yield %inserted_slice : tensor<4x5xf32>
    } -> (!secret.secret<tensor<4x5xf32>> {tensor_ext.layout = #layout})
    return %2 : !secret.secret<tensor<4x5xf32>>
  }
}
