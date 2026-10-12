// RUN: heir-opt --layout-propagation --split-input-file %s | FileCheck %s

// The stride-1 conv after a zero pad on the gapped result of a strided conv
// folds the pad into its read of the gapped data, so no conversion is needed.

// CHECK: @strided_then_padded_conv
func.func @strided_then_padded_conv(%arg0: !secret.secret<tensor<1x1x8x8xf32>>) -> !secret.secret<tensor<1x4x4x4xf32>> {
  %filter = arith.constant dense<2.500000e-01> : tensor<4x1x2x2xf32>
  %filter2 = arith.constant dense<1.000000e-01> : tensor<4x4x3x3xf32>
  %init = arith.constant dense<0.000000e+00> : tensor<1x4x4x4xf32>
  %zero = arith.constant 0.000000e+00 : f32

  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x8x8xf32>>) {
  ^body(%input0: tensor<1x1x8x8xf32>):
    // CHECK: %[[strided:[^ ]+]] = linalg.conv_2d_nchw_fchw
    // CHECK-SAME: strides = dense<2>
    // CHECK-NOT: tensor_ext.convert_layout
    // CHECK: tensor.pad %[[strided]]
    // CHECK-NOT: tensor_ext.convert_layout
    // CHECK: linalg.conv_2d_nchw_fchw
    // CHECK-SAME: heir.conv_folded_padding = 1
    // CHECK-SAME: gap_factor = 2
    %1 = linalg.conv_2d_nchw_fchw
      { dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
      ins(%input0, %filter : tensor<1x1x8x8xf32>, tensor<4x1x2x2xf32>)
      outs(%init : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    %padded = tensor.pad %1 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%i: index, %j: index, %k: index, %l: index):
      tensor.yield %zero : f32
    } : tensor<1x4x4x4xf32> to tensor<1x4x6x6xf32>
    %2 = linalg.conv_2d_nchw_fchw
      { dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
      ins(%padded, %filter2 : tensor<1x4x6x6xf32>, tensor<4x4x3x3xf32>)
      outs(%init : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    secret.yield %2 : tensor<1x4x4x4xf32>
  } -> !secret.secret<tensor<1x4x4x4xf32>>
  return %0 : !secret.secret<tensor<1x4x4x4xf32>>
}

// -----

// A 1-D conv only reads row-major data, so its gapped input is converted
// before the pad rather than after it, and the pad still folds.

// CHECK: @strided_then_padded_conv1d
func.func @strided_then_padded_conv1d(%arg0: !secret.secret<tensor<1x1x16xf32>>) -> !secret.secret<tensor<1x4x8xf32>> {
  %filter = arith.constant dense<2.500000e-01> : tensor<4x1x2xf32>
  %filter2 = arith.constant dense<1.000000e-01> : tensor<4x4x3xf32>
  %init = arith.constant dense<0.000000e+00> : tensor<1x4x8xf32>
  %zero = arith.constant 0.000000e+00 : f32

  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x16xf32>>) {
  ^body(%input0: tensor<1x1x16xf32>):
    // CHECK: %[[strided:[^ ]+]] = linalg.conv_1d_ncw_fcw
    // CHECK-SAME: strides = dense<2>
    // CHECK: %[[plain:[^ ]+]] = tensor_ext.convert_layout %[[strided]]
    // CHECK: tensor.pad %[[plain]]
    // CHECK: linalg.conv_1d_ncw_fcw
    // CHECK-SAME: heir.conv_folded_padding = 1
    // CHECK-SAME: gap_factor = 1
    %1 = linalg.conv_1d_ncw_fcw
      { dilations = dense<1> : tensor<1xi64>, strides = dense<2> : tensor<1xi64> }
      ins(%input0, %filter : tensor<1x1x16xf32>, tensor<4x1x2xf32>)
      outs(%init : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %padded = tensor.pad %1 low[0, 0, 1] high[0, 0, 1] {
    ^bb0(%i: index, %j: index, %k: index):
      tensor.yield %zero : f32
    } : tensor<1x4x8xf32> to tensor<1x4x10xf32>
    %2 = linalg.conv_1d_ncw_fcw
      { dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64> }
      ins(%padded, %filter2 : tensor<1x4x10xf32>, tensor<4x4x3xf32>)
      outs(%init : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    secret.yield %2 : tensor<1x4x8xf32>
  } -> !secret.secret<tensor<1x4x8xf32>>
  return %0 : !secret.secret<tensor<1x4x8xf32>>
}

// -----

// DropUnitDims leaves the pad at rank 3 between a collapse and an expand. It
// still folds into the read of the gapped data.

// CHECK: @strided_then_padded_conv_rank_reduced
// CHECK: %[[strided:[^ ]+]] = linalg.conv_2d_nchw_fchw
// CHECK-NOT: tensor_ext.convert_layout
// CHECK: tensor.collapse_shape %[[strided]]
// CHECK-NOT: tensor_ext.convert_layout
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-SAME: heir.conv_folded_padding = 1
// CHECK-SAME: gap_factor = 2

func.func @strided_then_padded_conv_rank_reduced(%arg0: !secret.secret<tensor<1x1x8x8xf32>>) -> !secret.secret<tensor<1x4x4x4xf32>> {
  %filter = arith.constant dense<2.500000e-01> : tensor<4x1x2x2xf32>
  %filter2 = arith.constant dense<1.000000e-01> : tensor<4x4x3x3xf32>
  %init = arith.constant dense<0.000000e+00> : tensor<1x4x4x4xf32>
  %zero = arith.constant 0.000000e+00 : f32
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x8x8xf32>>) {
  ^body(%input0: tensor<1x1x8x8xf32>):
    %1 = linalg.conv_2d_nchw_fchw
      { dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
      ins(%input0, %filter : tensor<1x1x8x8xf32>, tensor<4x1x2x2xf32>)
      outs(%init : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    %collapsed = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<1x4x4x4xf32> into tensor<4x4x4xf32>
    %padded = tensor.pad %collapsed low[0, 1, 1] high[0, 1, 1] {
    ^bb0(%i: index, %j: index, %k: index):
      tensor.yield %zero : f32
    } : tensor<4x4x4xf32> to tensor<4x6x6xf32>
    %expanded = tensor.expand_shape %padded [[0, 1], [2], [3]] output_shape [1, 4, 6, 6] : tensor<4x6x6xf32> into tensor<1x4x6x6xf32>
    %2 = linalg.conv_2d_nchw_fchw
      { dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
      ins(%expanded, %filter2 : tensor<1x4x6x6xf32>, tensor<4x4x3x3xf32>)
      outs(%init : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    secret.yield %2 : tensor<1x4x4x4xf32>
  } -> !secret.secret<tensor<1x4x4x4xf32>>
  return %0 : !secret.secret<tensor<1x4x4x4xf32>>
}

// -----

// The rank-reduced 1-D form converts before the pad, like the rank-3 one.

// CHECK: @strided_then_padded_conv1d_rank_reduced
// CHECK: linalg.conv_1d_ncw_fcw
// CHECK: %[[collapsed:[^ ]+]] = tensor.collapse_shape
// CHECK: %[[plain:[^ ]+]] = tensor_ext.convert_layout %[[collapsed]]
// CHECK: tensor.pad %[[plain]]
// CHECK: linalg.conv_1d_ncw_fcw
// CHECK-SAME: heir.conv_folded_padding = 1
// CHECK-SAME: gap_factor = 1

func.func @strided_then_padded_conv1d_rank_reduced(%arg0: !secret.secret<tensor<1x1x16xf32>>) -> !secret.secret<tensor<1x4x8xf32>> {
  %filter = arith.constant dense<2.500000e-01> : tensor<4x1x2xf32>
  %filter2 = arith.constant dense<1.000000e-01> : tensor<4x4x3xf32>
  %init = arith.constant dense<0.000000e+00> : tensor<1x4x8xf32>
  %zero = arith.constant 0.000000e+00 : f32
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x16xf32>>) {
  ^body(%input0: tensor<1x1x16xf32>):
    %1 = linalg.conv_1d_ncw_fcw
      { dilations = dense<1> : tensor<1xi64>, strides = dense<2> : tensor<1xi64> }
      ins(%input0, %filter : tensor<1x1x16xf32>, tensor<4x1x2xf32>)
      outs(%init : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %collapsed = tensor.collapse_shape %1 [[0, 1], [2]] : tensor<1x4x8xf32> into tensor<4x8xf32>
    %padded = tensor.pad %collapsed low[0, 1] high[0, 1] {
    ^bb0(%i: index, %j: index):
      tensor.yield %zero : f32
    } : tensor<4x8xf32> to tensor<4x10xf32>
    %expanded = tensor.expand_shape %padded [[0, 1], [2]] output_shape [1, 4, 10] : tensor<4x10xf32> into tensor<1x4x10xf32>
    %2 = linalg.conv_1d_ncw_fcw
      { dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64> }
      ins(%expanded, %filter2 : tensor<1x4x10xf32>, tensor<4x4x3xf32>)
      outs(%init : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    secret.yield %2 : tensor<1x4x8xf32>
  } -> !secret.secret<tensor<1x4x8xf32>>
  return %0 : !secret.secret<tensor<1x4x8xf32>>
}
