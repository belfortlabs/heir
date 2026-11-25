// RUN: heir-opt --strided-layout-propagation=min-slot-count=1024 %s | FileCheck %s

// A tensor that needs two ciphertexts: the channel planes of 60 elements are
// placed at a stride of 64, so sixteen of them fill a ciphertext.
// CHECK-DAG: #[[planes:.*]] = #tensor_ext.strided_layout<shape = [20, 6, 10], slots = 1024, digits = [(0, 1, 0, 64), (1, 1, 0, 10), (2, 1, 0, 1)]>
// The sum of a plane stays at the plane's first position.
// CHECK-DAG: #[[sums:.*]] = #tensor_ext.strided_layout<shape = [20], slots = 1024, digits = [(0, 1, 0, 64)]>

// CHECK: func.func @channel_scale
// CHECK-SAME: tensor_ext.layout = #[[planes]]
// CHECK: linalg.reduce
// CHECK-SAME: tensor_ext.layout = #[[sums]]
// The broadcast is produced in the layout of the tensor it multiplies, so
// the product needs no conversion.
// CHECK: linalg.broadcast
// CHECK-SAME: tensor_ext.layout = #[[planes]]
// CHECK-NOT: tensor_ext.convert_layout
// CHECK: arith.mulf
// CHECK-SAME: tensor_ext.layout = #[[planes]]
func.func @channel_scale(%x: !secret.secret<tensor<20x6x10xf32>>) -> !secret.secret<tensor<20x6x10xf32>> {
  %init = arith.constant dense<0.0> : tensor<20xf32>
  %empty = tensor.empty() : tensor<20x6x10xf32>
  %0 = secret.generic(%x: !secret.secret<tensor<20x6x10xf32>>) {
  ^body(%x0: tensor<20x6x10xf32>):
    %sums = linalg.reduce { arith.addf } ins(%x0 : tensor<20x6x10xf32>) outs(%init : tensor<20xf32>) dimensions = [1, 2]
    %scale = linalg.broadcast ins(%sums : tensor<20xf32>) outs(%empty : tensor<20x6x10xf32>) dimensions = [1, 2]
    %out = arith.mulf %x0, %scale : tensor<20x6x10xf32>
    secret.yield %out : tensor<20x6x10xf32>
  } -> !secret.secret<tensor<20x6x10xf32>>
  return %0 : !secret.secret<tensor<20x6x10xf32>>
}
