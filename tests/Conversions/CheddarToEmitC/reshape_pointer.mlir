// RUN: heir-opt --cheddar-to-emitc %s | FileCheck --implicit-check-not=memref.collapse_shape --implicit-check-not=memref.expand_shape %s

// A flat constant loaded as tensor<N> and expanded to the packed 1xN shape,
// multiplied with a packed plaintext. Folding the unit dim leaves reshapes
// between the two shapes, which are the same flat pointer.

// CHECK: func.func @expand_then_mul
// CHECK: return
func.func private @pack(%arg0: tensor<1x4xf32>) -> tensor<1x16xf32> {
  %zero = arith.constant 0.0 : f32
  %0 = tensor.pad %arg0 low[0, 0] high[0, 12] {
  ^bb0(%i: index, %j: index):
    tensor.yield %zero : f32
  } : tensor<1x4xf32> to tensor<1x16xf32>
  return %0 : tensor<1x16xf32>
}

func.func @expand_then_mul(%flat: tensor<16xf32>, %small: tensor<1x4xf32>) -> tensor<1x16xf32> {
  %packed = call @pack(%small) : (tensor<1x4xf32>) -> tensor<1x16xf32>
  %0 = tensor.expand_shape %flat [[0, 1]] output_shape [1, 16] : tensor<16xf32> into tensor<1x16xf32>
  %1 = arith.mulf %0, %packed : tensor<1x16xf32>
  return %1 : tensor<1x16xf32>
}
