// RUN: heir-opt --layout-propagation=min-slot-count=1024 --convert-to-ciphertext-semantics=min-slot-count=1024 %s | FileCheck %s
// RUN: heir-opt --layout-propagation=min-slot-count=1024 --convert-to-ciphertext-semantics=min-slot-count=1024 %s | FileCheck %s --check-prefix=MASKS

// A secret per-channel scale broadcast into a feature map that spans two
// ciphertexts: channels 0-3 fill the first and channels 4-7 the second, 256
// slots per channel.

module attributes {backend.lattigo, scheme.ckks} {

// The scale is replicated every 8 slots, so channel c needs a shift of 0, 7,
// 6, 5 (first ciphertext) or 4, 3, 2, 1 (second ciphertext) to reach the
// first slot of its block. With giant step 2, one baby-step rotation, eight
// masks and four giant-step rotations build both ciphertexts, and 8
// rotate-and-adds on the 2-ciphertext tensor fill the 256-slot blocks.

// CHECK: @two_ciphertexts
// CHECK-NOT: tensor_ext.remap
// CHECK-COUNT-5: tensor_ext.rotate {{.*}} : tensor<1x1024xf32>
// CHECK: tensor.insert_slice {{.*}}[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> into tensor<2x1024xf32>
// CHECK: tensor.insert_slice {{.*}}[1, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> into tensor<2x1024xf32>
// CHECK-COUNT-8: tensor_ext.rotate {{.*}} : tensor<2x1024xf32>
// CHECK-NOT: tensor_ext.rotate
// CHECK: arith.mulf {{.*}} : tensor<2x1024xf32>
// CHECK: return {{.*}} : !secret.secret<tensor<2x1024xf32>>

// MASKS: @two_ciphertexts
// MASKS-COUNT-8: arith.mulf {{.*}} : tensor<1x1024xf32>
// MASKS-NOT: arith.mulf {{.*}} : tensor<1x1024xf32>
func.func @two_ciphertexts(%arg0: !secret.secret<tensor<1x8xf32>>, %arg1: !secret.secret<tensor<1x8x16x16xf32>>) -> !secret.secret<tensor<1x8x16x16xf32>> {
  %0 = secret.generic(%arg0: !secret.secret<tensor<1x8xf32>>, %arg1: !secret.secret<tensor<1x8x16x16xf32>>) {
  ^body(%scale: tensor<1x8xf32>, %x: tensor<1x8x16x16xf32>):
    %empty = tensor.empty() : tensor<1x8x16x16xf32>
    %bcast = linalg.broadcast ins(%scale : tensor<1x8xf32>) outs(%empty : tensor<1x8x16x16xf32>) dimensions = [2, 3]
    %1 = arith.mulf %bcast, %x : tensor<1x8x16x16xf32>
    secret.yield %1 : tensor<1x8x16x16xf32>
  } -> !secret.secret<tensor<1x8x16x16xf32>>
  return %0 : !secret.secret<tensor<1x8x16x16xf32>>
}

}
