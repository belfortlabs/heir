// RUN: heir-opt --tensor-linalg-to-affine-loops %s | FileCheck %s

// A linalg.index becomes the induction variable of its loop. torch-mlir flips
// a ConvTranspose kernel this way: it reads the weight at mirrored indices.

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK: func @flip
func.func @flip(%weight: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %c1 = arith.constant 1 : index
  %init = arith.constant dense<0.000000e+00> : tensor<3x2xf32>
  // CHECK: affine.for [[i:%[^ ]*]] = 0 to 3
  // CHECK: affine.for [[j:%[^ ]*]] = 0 to 2
  // CHECK-NOT: linalg.index
  // CHECK: [[mirrored:%[^ ]*]] = arith.subi %{{.*}}, [[j]]
  // CHECK: tensor.extract %{{.*}}[[[mirrored]], [[i]]]
  %0 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%init : tensor<3x2xf32>) {
  ^bb0(%out: f32):
    %i = linalg.index 0 : index
    %j = linalg.index 1 : index
    %mirrored = arith.subi %c1, %j : index
    %x = tensor.extract %weight[%mirrored, %i] : tensor<2x3xf32>
    linalg.yield %x : f32
  } -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}
