// RUN: heir-opt --strided-layout-propagation=min-slot-count=16 \
// RUN:   --convert-to-ciphertext-semantics=min-slot-count=16 %s | FileCheck %s

// A 4x8 matrix times a secret vector of 8 in 16 slots. The result is laid out
// with period 4, so shifts are taken modulo 4: four diagonals instead of
// eleven. The transform writes each term to one copy of its output element,
// and two rotations complete the four copies.

// CHECK: func.func @matvec
// CHECK-SAME: !secret.secret<tensor<1x16xf32>>
// CHECK: %[[diagonals:.*]] = scf.for
// CHECK: %[[y:.*]] = kernel.block_linear_transform %{{.*}}, %[[diagonals]]
// CHECK-SAME: output_ciphertexts = [0, 0, 0, 0]
// CHECK-SAME: input_ciphertexts = [0, 0, 0, 0]
// CHECK-SAME: shifts = [0, 1, 2, 3]
// CHECK-SAME: tensor<1x16xf32>, tensor<4x16xf32> -> tensor<1x16xf32>
// CHECK: %[[r1:.*]] = tensor_ext.rotate %[[y]], %[[c4:.*]] :
// CHECK: %[[s1:.*]] = arith.addf %[[y]], %[[r1]]
// CHECK: %[[r2:.*]] = tensor_ext.rotate %[[s1]], %[[c8:.*]] :
// CHECK: %[[s2:.*]] = arith.addf %[[s1]], %[[r2]]
// CHECK: secret.yield %[[s2]]
func.func @matvec(%x: !secret.secret<tensor<8xf32>>, %m: tensor<4x8xf32>) -> !secret.secret<tensor<4xf32>> {
  %init = arith.constant dense<0.0> : tensor<4xf32>
  %0 = secret.generic(%x: !secret.secret<tensor<8xf32>>) {
  ^body(%input0: tensor<8xf32>):
    %1 = linalg.matvec ins(%m, %input0 : tensor<4x8xf32>, tensor<8xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
    secret.yield %1 : tensor<4xf32>
  } -> !secret.secret<tensor<4xf32>>
  return %0 : !secret.secret<tensor<4xf32>>
}
