// RUN: heir-opt %s --convert-to-ciphertext-semantics --split-input-file | FileCheck %s

#layout5 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 1024 = 0 and 0 <= i1 <= 783 and 0 <= slot <= 1023 }">
#layout6 = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 1024 = 0 and 0 <= i0 <= 783 and 0 <= slot <= 1023 }">
module {
  // CHECK: func.func @main
  func.func @main(%arg4: !secret.secret<tensor<1x784xf32>> {tensor_ext.layout = #layout5}) -> (!secret.secret<tensor<784xf32>> {tensor_ext.layout = #layout6}) {
    // CHECK: secret.generic
    // CHECK-NEXT: ^body(%[[input0:.*]]: tensor<1x1024xf32>)
    // CHECK: secret.yield %[[input0]]
    %7 = secret.generic(%arg4: !secret.secret<tensor<1x784xf32>> {tensor_ext.layout = #layout5}) {
    ^body(%input0: tensor<1x784xf32>):
      %collapsed = tensor.collapse_shape %input0 [[0, 1]] {tensor_ext.layout = #layout6} : tensor<1x784xf32> into tensor<784xf32>
      secret.yield %collapsed : tensor<784xf32>
    } -> (!secret.secret<tensor<784xf32>> {tensor_ext.layout = #layout6})
    return %7 : !secret.secret<tensor<784xf32>>
  }
}
