// RUN: heir-opt --strided-layout-propagation=min-slot-count=1024 %s | FileCheck %s

// A padded 3x3 convolution on 4 channels of 4x4. The pad is folded into the
// convolution: it and the reshape around it keep the layout of the unpadded
// tensor. The result stays on the input's grid.

// CHECK: #[[in:.*]] = #tensor_ext.strided_layout<shape = [1, 4, 4, 4], slots = 1024, digits = [(1, 1, 0, 16), (2, 1, 0, 4), (3, 1, 0, 1)]>
// CHECK: #[[collapsed:.*]] = #tensor_ext.strided_layout<shape = [4, 4, 4], slots = 1024, digits = [(0, 1, 0, 16), (1, 1, 0, 4), (2, 1, 0, 1)]>
// CHECK: func.func @conv
// CHECK-SAME: tensor_ext.layout = #[[in]]
// CHECK-SAME: tensor_ext.layout = #[[in]]
// CHECK: tensor.collapse_shape
// CHECK-SAME: tensor_ext.layout = #[[collapsed]]
// CHECK: tensor.pad
// CHECK: tensor_ext.layout = #[[collapsed]]
// CHECK: tensor.expand_shape
// CHECK-SAME: tensor_ext.layout = #[[collapsed]]
// CHECK: tensor_ext.assign_layout
// CHECK-SAME: layout = #[[in]]
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-SAME: tensor_ext.layout = #[[in]]
func.func @conv(%arg0: !secret.secret<tensor<1x4x4x4xf32>>, %filter: tensor<4x4x3x3xf32>, %bias: tensor<1x4x4x4xf32>) -> !secret.secret<tensor<1x4x4x4xf32>> {
  %zero = arith.constant 0.000000e+00 : f32
  %0 = secret.generic(%arg0: !secret.secret<tensor<1x4x4x4xf32>>) {
  ^body(%input0: tensor<1x4x4x4xf32>):
    %collapsed = tensor.collapse_shape %input0 [[0, 1], [2], [3]] : tensor<1x4x4x4xf32> into tensor<4x4x4xf32>
    %padded = tensor.pad %collapsed low[0, 1, 1] high[0, 1, 1] {
    ^bb0(%i: index, %j: index, %k: index):
      tensor.yield %zero : f32
    } : tensor<4x4x4xf32> to tensor<4x6x6xf32>
    %expanded = tensor.expand_shape %padded [[0, 1], [2], [3]] output_shape [1, 4, 6, 6] : tensor<4x6x6xf32> into tensor<1x4x6x6xf32>
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%expanded, %filter : tensor<1x4x6x6xf32>, tensor<4x4x3x3xf32>) outs(%bias : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    secret.yield %1 : tensor<1x4x4x4xf32>
  } -> !secret.secret<tensor<1x4x4x4xf32>>
  return %0 : !secret.secret<tensor<1x4x4x4xf32>>
}
