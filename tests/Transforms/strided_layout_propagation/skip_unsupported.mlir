// RUN: heir-opt --strided-layout-propagation=skip-unsupported=true --split-input-file %s 2>&1 | FileCheck %s

// A module with an operation outside the supported set is left as it was,
// including the division that would otherwise become a multiplication.

// CHECK-NOT: error
// CHECK: func.func @two_secret_inputs
// CHECK-NOT: tensor_ext.layout
// CHECK-NOT: tensor_ext.assign_layout
// CHECK: arith.divf
// CHECK: linalg.matmul
// CHECK-NOT: tensor_ext.layout
func.func @two_secret_inputs(%a: !secret.secret<tensor<4x4xf32>>, %b: !secret.secret<tensor<4x4xf32>>) -> !secret.secret<tensor<4x4xf32>> {
  %init = arith.constant dense<0.0> : tensor<4x4xf32>
  %two = arith.constant dense<2.0> : tensor<4x4xf32>
  %0 = secret.generic(%a: !secret.secret<tensor<4x4xf32>>, %b: !secret.secret<tensor<4x4xf32>>) {
  ^body(%x: tensor<4x4xf32>, %y: tensor<4x4xf32>):
    %half = arith.divf %x, %two : tensor<4x4xf32>
    %1 = linalg.matmul ins(%half, %y : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    secret.yield %1 : tensor<4x4xf32>
  } -> !secret.secret<tensor<4x4xf32>>
  return %0 : !secret.secret<tensor<4x4xf32>>
}

// -----

// A secret scalar has no strided layout.
// CHECK: func.func @scalar
// CHECK-NOT: tensor_ext.layout
func.func @scalar(%a: !secret.secret<f32>) -> !secret.secret<f32> {
  %0 = secret.generic(%a: !secret.secret<f32>) {
  ^body(%x: f32):
    %1 = arith.addf %x, %x : f32
    secret.yield %1 : f32
  } -> !secret.secret<f32>
  return %0 : !secret.secret<f32>
}

// -----

// A rotation is not an elementwise operation of the tensor it is applied to.
// CHECK: func.func @rotation
// CHECK-NOT: tensor_ext.layout
func.func @rotation(%a: !secret.secret<tensor<8xf32>>) -> !secret.secret<tensor<8xf32>> {
  %0 = secret.generic(%a: !secret.secret<tensor<8xf32>>) {
  ^body(%x: tensor<8xf32>):
    %shift = arith.constant 1 : index
    %1 = tensor_ext.rotate %x, %shift : tensor<8xf32>, index
    secret.yield %1 : tensor<8xf32>
  } -> !secret.secret<tensor<8xf32>>
  return %0 : !secret.secret<tensor<8xf32>>
}
