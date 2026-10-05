// RUN: heir-opt %s --split-input-file \
// RUN:   --convert-to-ciphertext-semantics=min-slot-count=32 \
// RUN:   --mlir-print-elementsattrs-with-hex-if-larger=1 | FileCheck %s

// A dense resource filter with a list of layouts (the diagonalized Toeplitz
// steps of a 2x2 conv filter) folds once through the composed layout into a
// packed constant, without materializing the intermediate tensors. The same
// filter as a small dense constant folds layout by layout, and both give the
// same packed constant.

#layout3 = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot, o2, o3, o4] : i0 = 0 and i1 = 0 and ct = 0 and slot = 0 and o4 = 4i2 + i3 + 4o2 + o3 and 0 <= i2 <= 1 and 0 <= i3 <= 1 and 0 <= o2 <= 2 and o3 >= 0 and -i3 <= o3 <= 3 - i3 and o3 <= 2 }">
#layout4 = #tensor_ext.layout<"{ [i0, i1, i2, i3, i4] -> [ct, slot, o2] : i0 = 0 and i1 = 0 and ct = 3i2 + i3 and slot = 0 and o2 = i4 and 0 <= i2 <= 2 and 0 <= i3 <= 2 and 0 <= i4 <= 15 }">
#layout5 = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i1 = 0 and ct = i0 and slot = i2 and 0 <= i0 <= 8 and 0 <= i2 <= 15 }">
#layout6 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (i0 - i1 + ct) mod 16 = 0 and (-i0 + slot) mod 16 = 0 and 0 <= i0 <= 8 and 0 <= i1 <= 15 and 0 <= ct <= 15 and 0 <= slot <= 31 }">
#layout7 = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and i1 = 0 and 0 <= i2 <= 1 and 0 <= i3 <= 1 and 0 <= ct <= 15 and 0 <= slot <= 31 and 16*floor((-3 - 4i2 - i3 + ct)/16) <= -16 - 4i2 - i3 + ct and 16*floor((7 + slot)/16) >= 45 + 12i2 + 4i3 - 3ct + slot + 48*floor((-3 - 4i2 - i3 + ct)/16) and 16*floor((7 + slot)/16) >= 46 + 12i2 + 3i3 - 3ct + slot + 48*floor((-3 - 4i2 - i3 + ct)/16) and 16*floor((7 + slot)/16) >= -31 + ct + slot - 16*floor((-3 - 4i2 - i3 + ct)/16) and 16*floor((7 + slot)/16) <= slot and 16*floor((7 + slot)/16) <= -16 + ct + slot - 16*floor((-3 - 4i2 - i3 + ct)/16) and 16*floor((7 + slot)/16) <= 48 + 12i2 + 3i3 - 3ct + slot + 48*floor((-3 - 4i2 - i3 + ct)/16) and 16*floor((7 + slot)/16) <= 48 + 12i2 + 4i3 - 3ct + slot + 48*floor((-3 - 4i2 - i3 + ct)/16) }">
module {
  // CHECK: func.func private @_assign_layout
  // CHECK-NOT: scf.for
  // CHECK: arith.constant dense<"0x[[DATA:[0-9A-F]+]]"> : tensor<16x32xf32>
  // CHECK: func.func @filter_by_steps
  func.func @filter_by_steps() -> (!secret.secret<tensor<1x1x2x2xf32>> {tensor_ext.layout = #layout7}) {
    %cst = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0]]]]> : tensor<1x1x2x2xf32>
    %0 = secret.generic() {
      %1 = tensor_ext.assign_layout %cst {layout = [#layout3, #layout4, #layout5, #layout6], tensor_ext.layout = #layout7} : tensor<1x1x2x2xf32>
      secret.yield %1 : tensor<1x1x2x2xf32>
    } -> (!secret.secret<tensor<1x1x2x2xf32>> {tensor_ext.layout = #layout7})
    return %0 : !secret.secret<tensor<1x1x2x2xf32>>
  }
}

// -----

#layout3 = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot, o2, o3, o4] : i0 = 0 and i1 = 0 and ct = 0 and slot = 0 and o4 = 4i2 + i3 + 4o2 + o3 and 0 <= i2 <= 1 and 0 <= i3 <= 1 and 0 <= o2 <= 2 and o3 >= 0 and -i3 <= o3 <= 3 - i3 and o3 <= 2 }">
#layout4 = #tensor_ext.layout<"{ [i0, i1, i2, i3, i4] -> [ct, slot, o2] : i0 = 0 and i1 = 0 and ct = 3i2 + i3 and slot = 0 and o2 = i4 and 0 <= i2 <= 2 and 0 <= i3 <= 2 and 0 <= i4 <= 15 }">
#layout5 = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i1 = 0 and ct = i0 and slot = i2 and 0 <= i0 <= 8 and 0 <= i2 <= 15 }">
#layout6 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (i0 - i1 + ct) mod 16 = 0 and (-i0 + slot) mod 16 = 0 and 0 <= i0 <= 8 and 0 <= i1 <= 15 and 0 <= ct <= 15 and 0 <= slot <= 31 }">
#layout7 = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and i1 = 0 and 0 <= i2 <= 1 and 0 <= i3 <= 1 and 0 <= ct <= 15 and 0 <= slot <= 31 and 16*floor((-3 - 4i2 - i3 + ct)/16) <= -16 - 4i2 - i3 + ct and 16*floor((7 + slot)/16) >= 45 + 12i2 + 4i3 - 3ct + slot + 48*floor((-3 - 4i2 - i3 + ct)/16) and 16*floor((7 + slot)/16) >= 46 + 12i2 + 3i3 - 3ct + slot + 48*floor((-3 - 4i2 - i3 + ct)/16) and 16*floor((7 + slot)/16) >= -31 + ct + slot - 16*floor((-3 - 4i2 - i3 + ct)/16) and 16*floor((7 + slot)/16) <= slot and 16*floor((7 + slot)/16) <= -16 + ct + slot - 16*floor((-3 - 4i2 - i3 + ct)/16) and 16*floor((7 + slot)/16) <= 48 + 12i2 + 3i3 - 3ct + slot + 48*floor((-3 - 4i2 - i3 + ct)/16) and 16*floor((7 + slot)/16) <= 48 + 12i2 + 4i3 - 3ct + slot + 48*floor((-3 - 4i2 - i3 + ct)/16) }">
module {
  // CHECK: func.func @filter
  // CHECK-NOT: scf.for
  // CHECK: %[[PACKED:.*]] = arith.constant dense_resource<filter_packed> : tensor<16x32xf32>
  // CHECK: secret.yield %[[PACKED]]
  // CHECK: filter_packed: "0x04000000[[DATA]]"
  func.func @filter() -> (!secret.secret<tensor<1x1x2x2xf32>> {tensor_ext.layout = #layout7}) {
    %cst = arith.constant dense_resource<filter> : tensor<1x1x2x2xf32>
    %0 = secret.generic() {
      %1 = tensor_ext.assign_layout %cst {layout = [#layout3, #layout4, #layout5, #layout6], tensor_ext.layout = #layout7} : tensor<1x1x2x2xf32>
      secret.yield %1 : tensor<1x1x2x2xf32>
    } -> (!secret.secret<tensor<1x1x2x2xf32>> {tensor_ext.layout = #layout7})
    return %0 : !secret.secret<tensor<1x1x2x2xf32>>
  }
}

{-#
  dialect_resources: {
    builtin: {
      filter: "0x040000000000803F000000400000404000008040"
    }
  }
#-}
