// RUN: heir-opt --layout-propagation=min-slot-count=16 --split-input-file %s | FileCheck %s

// CHECK: #[[layout:.*]] = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-4i0 - i1 + slot) mod 32 = 0 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= slot <= 15 }">

module {
  // CHECK: func.func @main
  // CHECK: tensor.extract_slice
  // CHECK-SAME: {heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 4, 4>}, tensor_ext.layout = #[[layout]]}
  // CHECK: return
  func.func @main(%arg0: !secret.secret<tensor<1x2x4x4xf32>>) -> !secret.secret<tensor<4x4xf32>> {
    %2 = secret.generic(%arg0: !secret.secret<tensor<1x2x4x4xf32>>) {
    ^body(%input0: tensor<1x2x4x4xf32>):
      %extracted_slice = tensor.extract_slice %input0[0, 0, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : tensor<1x2x4x4xf32> to tensor<4x4xf32>
      secret.yield %extracted_slice : tensor<4x4xf32>
    } -> !secret.secret<tensor<4x4xf32>>
    return %2 : !secret.secret<tensor<4x4xf32>>
  }
}

// -----

// CHECK: #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-4i0 - i1 + slot) mod 32 = 0 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= slot <= 15 }">

module {
  // CHECK: func.func @offset
  // CHECK: tensor.extract_slice
  // CHECK-SAME: {heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 4, 4>}, tensor_ext.layout = #[[layout]]}
  // CHECK: return
  func.func @offset(%arg0: !secret.secret<tensor<2x1x4x4xf32>>) -> !secret.secret<tensor<4x4xf32>> {
    %2 = secret.generic(%arg0: !secret.secret<tensor<2x1x4x4xf32>>) {
    ^body(%input0: tensor<2x1x4x4xf32>):
      %extracted_slice = tensor.extract_slice %input0[1, 0, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : tensor<2x1x4x4xf32> to tensor<4x4xf32>
      secret.yield %extracted_slice : tensor<4x4xf32>
    } -> !secret.secret<tensor<4x4xf32>>
    return %2 : !secret.secret<tensor<4x4xf32>>
  }
}

// -----

// Rank-preserving slice: the unit batch dimension stays in the layout and i3 is
// tied to the slot.

// CHECK: #[[sliced:.*]] = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and (-4 - 8i2 - i3 + slot + 16*floor((i2)/2)) mod 64 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 3 and 0 <= i3 <= 3 and -1 + 4i1 + i2 <= 2ct <= 4i1 + i2 and 0 <= slot <= 15 }">
// CHECK: func.func @keep_unit_dim
// CHECK: tensor.extract_slice
// CHECK-SAME: tensor_ext.layout = #[[sliced]]}

module {
  func.func @keep_unit_dim(%arg0: !secret.secret<tensor<1x2x4x8xf32>>) -> !secret.secret<tensor<1x2x4x4xf32>> {
    %2 = secret.generic(%arg0: !secret.secret<tensor<1x2x4x8xf32>>) {
    ^body(%input0: tensor<1x2x4x8xf32>):
      %extracted_slice = tensor.extract_slice %input0[0, 0, 0, 4] [1, 2, 4, 4] [1, 1, 1, 1] : tensor<1x2x4x8xf32> to tensor<1x2x4x4xf32>
      secret.yield %extracted_slice : tensor<1x2x4x4xf32>
    } -> !secret.secret<tensor<1x2x4x4xf32>>
    return %2 : !secret.secret<tensor<1x2x4x4xf32>>
  }
}
