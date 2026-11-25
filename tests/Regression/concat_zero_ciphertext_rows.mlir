// RUN: heir-opt "--annotate-module=backend=lattigo scheme=ckks" --torch-linalg-to-ckks=min-slot-count=16 %s | FileCheck %s
// RUN: heir-opt "--annotate-module=backend=openfhe scheme=ckks" --torch-linalg-to-ckks=min-slot-count=16 %s | FileCheck %s

// Each channel is one ciphertext, and each operand of the concatenation is
// moved by a linear transform that writes only some of the result's
// ciphertexts. The other rows of the transform's result are zero. They are
// cleartext rows in a tensor of ciphertexts and are trivially encrypted, which
// is an addition to an encryption of zero that the client supplies.

// CHECK: func.func @concat
// CHECK: lwe.radd_plain
// CHECK: return
// CHECK: @concat__encrypt__zero__0
func.func @concat(%x: tensor<1x2x4x4xf32> {secret.secret}, %y: tensor<1x3x4x4xf32> {secret.secret}) -> tensor<1x5x4x4xf32> {
  %out = tensor.concat dim(1) %x, %y : (tensor<1x2x4x4xf32>, tensor<1x3x4x4xf32>) -> tensor<1x5x4x4xf32>
  return %out : tensor<1x5x4x4xf32>
}
