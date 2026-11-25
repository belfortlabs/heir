// RUN: heir-opt --strided-layout-propagation=min-slot-count=1024 %s | FileCheck %s

// A stride-2 convolution leaves its result on every second position of the
// input's grid, with the four output channels in the positions between. The
// second secret argument is row-major, so it is converted to that layout
// before the sum, and the cleartext constant is packed into it.

// CHECK-DAG: #[[row_major:.*]] = #tensor_ext.strided_layout<shape = [1, 4, 4, 4], slots = 1024, digits = [(1, 1, 0, 16), (2, 1, 0, 4), (3, 1, 0, 1)]>
// CHECK-DAG: #[[gapped:.*]] = #tensor_ext.strided_layout<shape = [1, 4, 4, 4], slots = 1024, digits = [(1, 1, 2, 1), (1, 2, 0, 8), (2, 1, 0, 16), (3, 1, 0, 2)]>
// CHECK: func.func @join
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-SAME: tensor_ext.layout = #[[gapped]]
// CHECK: tensor_ext.convert_layout
// CHECK-SAME: from_layout = #[[row_major]]
// CHECK-SAME: to_layout = #[[gapped]]
// CHECK: arith.addf
// CHECK-SAME: tensor_ext.layout = #[[gapped]]
// CHECK: tensor_ext.assign_layout
// CHECK-SAME: layout = #[[gapped]]
// CHECK: arith.mulf
// CHECK-SAME: tensor_ext.layout = #[[gapped]]
func.func @join(%x: !secret.secret<tensor<1x2x8x8xf32>>, %y: !secret.secret<tensor<1x4x4x4xf32>>, %w: tensor<4x2x2x2xf32>) -> !secret.secret<tensor<1x4x4x4xf32>> {
  %init = arith.constant dense<0.0> : tensor<1x4x4x4xf32>
  %half = arith.constant dense<0.5> : tensor<1x4x4x4xf32>
  %0 = secret.generic(%x: !secret.secret<tensor<1x2x8x8xf32>>, %y: !secret.secret<tensor<1x4x4x4xf32>>) {
  ^body(%x0: tensor<1x2x8x8xf32>, %y0: tensor<1x4x4x4xf32>):
    %h = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%x0, %w : tensor<1x2x8x8xf32>, tensor<4x2x2x2xf32>) outs(%init : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    %sum = arith.addf %h, %y0 : tensor<1x4x4x4xf32>
    %out = arith.mulf %sum, %half : tensor<1x4x4x4xf32>
    secret.yield %out : tensor<1x4x4x4xf32>
  } -> !secret.secret<tensor<1x4x4x4xf32>>
  return %0 : !secret.secret<tensor<1x4x4x4xf32>>
}
