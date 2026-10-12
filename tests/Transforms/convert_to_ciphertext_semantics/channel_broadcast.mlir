// RUN: heir-opt --layout-propagation=min-slot-count=1024 --convert-to-ciphertext-semantics=min-slot-count=1024 %s | FileCheck %s

// A secret per-channel scale broadcast over the spatial dims takes the layout
// of the feature map it is multiplied into, so neither operand of the
// multiplication needs a layout conversion.

module attributes {backend.lattigo, scheme.ckks} {

// The scale is replicated every 4 slots. Two masked rotations move each
// channel to the first slot of its 16-slot block, and 8 rotate-and-adds fill
// the blocks and the replicas of the 64-slot feature map.

// CHECK: @row_major
// CHECK-NOT: tensor_ext.remap
// CHECK-COUNT-10: tensor_ext.rotate %
// CHECK-NOT: tensor_ext.rotate
// CHECK: return
func.func @row_major(%arg0: !secret.secret<tensor<1x4xf32>>, %arg1: !secret.secret<tensor<1x4x4x4xf32>>) -> !secret.secret<tensor<1x4x4x4xf32>> {
  %0 = secret.generic(%arg0: !secret.secret<tensor<1x4xf32>>, %arg1: !secret.secret<tensor<1x4x4x4xf32>>) {
  ^body(%scale: tensor<1x4xf32>, %x: tensor<1x4x4x4xf32>):
    %empty = tensor.empty() : tensor<1x4x4x4xf32>
    %bcast = linalg.broadcast ins(%scale : tensor<1x4xf32>) outs(%empty : tensor<1x4x4x4xf32>) dimensions = [2, 3]
    %1 = arith.mulf %bcast, %x : tensor<1x4x4x4xf32>
    secret.yield %1 : tensor<1x4x4x4xf32>
  } -> !secret.secret<tensor<1x4x4x4xf32>>
  return %0 : !secret.secret<tensor<1x4x4x4xf32>>
}

// The feature map of a stride-2 convolution has a gapped layout.

// CHECK: @gapped
// CHECK: tensor_ext.rotate_and_reduce
// CHECK-NOT: tensor_ext.remap
// CHECK-COUNT-9: tensor_ext.rotate %
// CHECK-NOT: tensor_ext.rotate
// CHECK: return
func.func @gapped(%arg0: !secret.secret<tensor<1x4xf32>>, %arg1: !secret.secret<tensor<1x1x8x8xf32>>) -> !secret.secret<tensor<1x4x4x4xf32>> {
  %filter = arith.constant dense<[[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]]> : tensor<4x1x1x1xf32>
  %init = arith.constant dense<0.0> : tensor<1x4x4x4xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<1x4xf32>>, %arg1: !secret.secret<tensor<1x1x8x8xf32>>) {
  ^body(%scale: tensor<1x4xf32>, %input: tensor<1x1x8x8xf32>):
    %x = linalg.conv_2d_nchw_fchw
      { dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
      ins(%input, %filter : tensor<1x1x8x8xf32>, tensor<4x1x1x1xf32>)
      outs(%init : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    %empty = tensor.empty() : tensor<1x4x4x4xf32>
    %bcast = linalg.broadcast ins(%scale : tensor<1x4xf32>) outs(%empty : tensor<1x4x4x4xf32>) dimensions = [2, 3]
    %1 = arith.mulf %bcast, %x : tensor<1x4x4x4xf32>
    secret.yield %1 : tensor<1x4x4x4xf32>
  } -> !secret.secret<tensor<1x4x4x4xf32>>
  return %0 : !secret.secret<tensor<1x4x4x4xf32>>
}

}
