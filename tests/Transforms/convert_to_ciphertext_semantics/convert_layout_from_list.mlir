// RUN: heir-opt %s --convert-to-ciphertext-semantics=min-slot-count=32 | FileCheck %s

// A convert_layout whose from_layout is a list of steps (as a strided conv
// records its result layout) converts from the layout the list composes to,
// instead of being left unconverted.

#step1 = #tensor_ext.layout<"{ [i0] -> [o0, o1] : o0 = i0 and o1 = i0 and 0 <= i0 <= 3 }">
#step2 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = i0 and slot = i1 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= slot <= 31 }">
#composed = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = i0 and slot = i0 and 0 <= i0 <= 3 and 0 <= slot <= 31 }">
#target = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 4 = 0 and 0 <= i0 <= 3 and 0 <= slot <= 31 }">
module {
  // CHECK: func.func @convert_from_list
  // CHECK-NOT: tensor_ext.convert_layout
  // CHECK: secret.yield {{.*}} : tensor<1x32xi16>
  func.func @convert_from_list(%arg0: !secret.secret<tensor<4xi16>> {tensor_ext.layout = #composed}) -> (!secret.secret<tensor<4xi16>> {tensor_ext.layout = #target}) {
    %0 = secret.generic(%arg0: !secret.secret<tensor<4xi16>> {tensor_ext.layout = #composed}) {
    ^body(%input0: tensor<4xi16>):
      %1 = tensor_ext.convert_layout %input0 {from_layout = [#step1, #step2], tensor_ext.layout = #target, to_layout = #target} : tensor<4xi16>
      secret.yield %1 : tensor<4xi16>
    } -> (!secret.secret<tensor<4xi16>> {tensor_ext.layout = #target})
    return %0 : !secret.secret<tensor<4xi16>>
  }
}
