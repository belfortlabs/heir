// RUN: heir-opt --strided-layout-propagation=min-slot-count=1024 %s | FileCheck %s

// CHECK-DAG: #[[arg:.*]] = #tensor_ext.strided_layout<shape = [2, 3, 4], slots = 1024, digits = [(0, 1, 0, 12), (1, 1, 0, 4), (2, 1, 0, 1)]>
// CHECK-DAG: #[[transposed:.*]] = #tensor_ext.strided_layout<shape = [4, 2, 3], slots = 1024, digits = [(0, 1, 0, 1), (1, 1, 0, 12), (2, 1, 0, 4)]>
// CHECK-DAG: #[[rows:.*]] = #tensor_ext.strided_layout<shape = [4, 8], slots = 1024, digits = [(0, 1, 0, 16), (1, 1, 0, 1)]>
// CHECK-DAG: #[[flat:.*]] = #tensor_ext.strided_layout<shape = [24], slots = 1024, digits = [(0, 1, 6, 4), (0, 6, 0, 1)]>
// CHECK-DAG: #[[sliced:.*]] = #tensor_ext.strided_layout<shape = [6], slots = 1024, digits = [(0, 1, 0, 1)], period = 8>

// A transpose permutes the axes of the layout and moves no element: the
// result keeps the row-major positions of the argument.

// CHECK: func.func @transpose
// CHECK-NOT: tensor_ext.convert_layout
// CHECK: linalg.transpose
// CHECK-SAME: tensor_ext.layout = #[[transposed]]
func.func @transpose(%x: !secret.secret<tensor<2x3x4xf32>>) -> !secret.secret<tensor<4x2x3xf32>> {
  %init = tensor.empty() : tensor<4x2x3xf32>
  %0 = secret.generic(%x: !secret.secret<tensor<2x3x4xf32>>) {
  ^body(%x0: tensor<2x3x4xf32>):
    %t = linalg.transpose ins(%x0 : tensor<2x3x4xf32>) outs(%init : tensor<4x2x3xf32>) permutation = [2, 0, 1]
    secret.yield %t : tensor<4x2x3xf32>
  } -> !secret.secret<tensor<4x2x3xf32>>
  return %0 : !secret.secret<tensor<4x2x3xf32>>
}

// A slice that takes every second row and drops the unit axis in front.

// CHECK: func.func @slice
// CHECK: tensor.extract_slice
// CHECK-SAME: tensor_ext.layout = #[[rows]]
func.func @slice(%x: !secret.secret<tensor<1x8x8xf32>>) -> !secret.secret<tensor<4x8xf32>> {
  %0 = secret.generic(%x: !secret.secret<tensor<1x8x8xf32>>) {
  ^body(%x0: tensor<1x8x8xf32>):
    %s = tensor.extract_slice %x0[0, 0, 0] [1, 4, 8] [1, 2, 1] : tensor<1x8x8xf32> to tensor<4x8xf32>
    secret.yield %s : tensor<4x8xf32>
  } -> !secret.secret<tensor<4x8xf32>>
  return %0 : !secret.secret<tensor<4x8xf32>>
}

// Flattening the transposed tensor also moves nothing. The flat index splits
// into two digits: its remainder modulo 6 counts the positions of one value of
// the leading axis, at stride 4, and its quotient steps through that axis.

// CHECK: func.func @flatten_transposed
// CHECK: tensor.collapse_shape
// CHECK-SAME: tensor_ext.layout = #[[flat]]

// Every fourth element of that vector is not at regular positions, so the
// slice moves its elements into a new, row-major layout.

// CHECK: tensor.extract_slice
// CHECK-SAME: tensor_ext.layout = #[[sliced]]
func.func @flatten_transposed(%x: !secret.secret<tensor<2x3x4xf32>>) -> !secret.secret<tensor<6xf32>> {
  %init = tensor.empty() : tensor<4x2x3xf32>
  %0 = secret.generic(%x: !secret.secret<tensor<2x3x4xf32>>) {
  ^body(%x0: tensor<2x3x4xf32>):
    %t = linalg.transpose ins(%x0 : tensor<2x3x4xf32>) outs(%init : tensor<4x2x3xf32>) permutation = [2, 0, 1]
    %f = tensor.collapse_shape %t [[0, 1, 2]] : tensor<4x2x3xf32> into tensor<24xf32>
    %s = tensor.extract_slice %f[0] [6] [4] : tensor<24xf32> to tensor<6xf32>
    secret.yield %s : tensor<6xf32>
  } -> !secret.secret<tensor<6xf32>>
  return %0 : !secret.secret<tensor<6xf32>>
}
