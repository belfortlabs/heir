// RUN: heir-opt --layout-propagation=min-slot-count=1024 %s | FileCheck %s

// A stride-2 conv leaves its result in a pixel-shuffled, gapped packing. The
// second conv pads that result, so its data operand is neither row major nor
// unpadded. Both convs must still fold their pad into their own `padding`
// parameter, and neither may pay for an online layout conversion: the second
// conv reads the gapped packing where it sits by absorbing it into its
// plaintext diagonal filter.
//
// The 2-D layout is materialized as a sequence of relations, so absorbing adds
// a step to that sequence rather than composing into a single relation.

// CHECK-DAG: #[[$PACK1:.*]] = #tensor_ext.conv_packing<matrixDataType = tensor<1x2x4x4xf32>, padding = 1, interchangeRows = true, absorbedMatrixWidth = 0>
// CHECK-DAG: #[[$PACK2:.*]] = #tensor_ext.conv_packing<matrixDataType = tensor<1x2x2x2xf32>, padding = 1, interchangeRows = false, absorbedMatrixWidth = 1024>
// CHECK-NOT: tensor_ext.convert_layout
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-SAME: tensor_ext.conv_packing = #[[$PACK1]]
// CHECK-NOT: tensor_ext.convert_layout
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-SAME: tensor_ext.conv_packing = #[[$PACK2]]
// CHECK-NOT: tensor_ext.convert_layout

func.func @conv2d_gapped_folded_padding(
    %arg0: !secret.secret<tensor<1x2x4x4xf32>>) -> !secret.secret<tensor<1x2x2x2xf32>> {
  %out = arith.constant dense<0.000000e+00> : tensor<1x2x2x2xf32>
  %filter1 = arith.constant dense<2.000000e+00> : tensor<2x2x3x3xf32>
  %filter2 = arith.constant dense<3.000000e+00> : tensor<2x2x3x3xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x2x4x4xf32>>) {
  ^body(%input0: tensor<1x2x4x4xf32>):
    %zero = arith.constant 0.000000e+00 : f32
    %padded0 = tensor.pad %input0 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%i: index, %j: index, %k: index, %l: index):
      tensor.yield %zero : f32
    } : tensor<1x2x4x4xf32> to tensor<1x2x6x6xf32>
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded0, %filter1 : tensor<1x2x6x6xf32>, tensor<2x2x3x3xf32>) outs(%out : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %padded1 = tensor.pad %1 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%i: index, %j: index, %k: index, %l: index):
      tensor.yield %zero : f32
    } : tensor<1x2x2x2xf32> to tensor<1x2x4x4xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded1, %filter2 : tensor<1x2x4x4xf32>, tensor<2x2x3x3xf32>) outs(%out : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    secret.yield %2 : tensor<1x2x2x2xf32>
  } -> !secret.secret<tensor<1x2x2x2xf32>>
  return %0 : !secret.secret<tensor<1x2x2x2xf32>>
}
