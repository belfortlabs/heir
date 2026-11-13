// RUN: heir-opt --layout-propagation %s | FileCheck %s

// Each function feeds a slice of a strided conv's gapped result to a kernel
// that makes a new layout (row-major, per-row, bicyclic, tricyclic or the
// default layout), so the kernel's result is not gapped and gets kernel info
// with gap factor 1. A conv that follows must read it as ungapped data. An
// elementwise op keeps its operand's layout, so its result stays gapped. The
// first three functions reach a different matmul kernel each.

// CHECK-DAG: name = "MatmulDiagonal"
// CHECK-DAG: name = "MatmulBicyclicDiagonal"
// CHECK-DAG: name = "MatmulBicyclic"

// CHECK: @strided_conv_per_row_matmul_conv
// CHECK: linalg.matmul
// CHECK-SAME: heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 4, 4>}
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-SAME: heir.kernel_info = {gap_factor = 1 : i64, input_shape = array<i64: 1, 1, 4, 4>, result_shape = array<i64: 1, 2, 4, 4>}
func.func @strided_conv_per_row_matmul_conv(%arg0: !secret.secret<tensor<1x1x8x8xf32>>) -> !secret.secret<tensor<1x2x4x4xf32>> {
  %filter = arith.constant dense<2.500000e-01> : tensor<4x1x2x2xf32>
  %bias = arith.constant dense<1.000000e+00> : tensor<1x4x4x4xf32>
  %matrix = arith.constant dense<6.250000e-02> : tensor<4x4xf32>
  %init = arith.constant dense<0.000000e+00> : tensor<4x4xf32>
  %fc = arith.constant dense<5.000000e-01> : tensor<2x1x1x1xf32>
  %fc_bias = arith.constant dense<1.000000e+00> : tensor<1x2x4x4xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x8x8xf32>>) {
  ^body(%input0: tensor<1x1x8x8xf32>):
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%input0, %filter : tensor<1x1x8x8xf32>, tensor<4x1x2x2xf32>) outs(%bias : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    %2 = tensor.extract_slice %1[0, 0, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : tensor<1x4x4x4xf32> to tensor<4x4xf32>
    %3 = linalg.matmul ins(%2, %matrix : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    %4 = tensor.expand_shape %3 [[0, 1, 2], [3]] output_shape [1, 1, 4, 4] : tensor<4x4xf32> into tensor<1x1x4x4xf32>
    %5 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%4, %fc : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>) outs(%fc_bias : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    secret.yield %5 : tensor<1x2x4x4xf32>
  } -> !secret.secret<tensor<1x2x4x4xf32>>
  return %0 : !secret.secret<tensor<1x2x4x4xf32>>
}

// Coprime dims select the bicyclic kernel with a diagonal plaintext weight.
// CHECK: @strided_conv_bicyclic_diagonal_matmul_conv
// CHECK: linalg.matmul
// CHECK-SAME: heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 5, 3>}
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-SAME: heir.kernel_info = {gap_factor = 1 : i64, input_shape = array<i64: 1, 1, 5, 3>, result_shape = array<i64: 1, 2, 5, 3>}
func.func @strided_conv_bicyclic_diagonal_matmul_conv(%arg0: !secret.secret<tensor<1x1x10x10xf32>>) -> !secret.secret<tensor<1x2x5x3xf32>> {
  %filter = arith.constant dense<2.500000e-01> : tensor<4x1x2x2xf32>
  %bias = arith.constant dense<1.000000e+00> : tensor<1x4x5x5xf32>
  %matrix = arith.constant dense<6.250000e-02> : tensor<4x3xf32>
  %init = arith.constant dense<0.000000e+00> : tensor<5x3xf32>
  %fc = arith.constant dense<5.000000e-01> : tensor<2x1x1x1xf32>
  %fc_bias = arith.constant dense<1.000000e+00> : tensor<1x2x5x3xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x10x10xf32>>) {
  ^body(%input0: tensor<1x1x10x10xf32>):
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%input0, %filter : tensor<1x1x10x10xf32>, tensor<4x1x2x2xf32>) outs(%bias : tensor<1x4x5x5xf32>) -> tensor<1x4x5x5xf32>
    %2 = tensor.extract_slice %1[0, 0, 0, 0] [1, 1, 5, 4] [1, 1, 1, 1] : tensor<1x4x5x5xf32> to tensor<5x4xf32>
    %3 = linalg.matmul ins(%2, %matrix : tensor<5x4xf32>, tensor<4x3xf32>) outs(%init : tensor<5x3xf32>) -> tensor<5x3xf32>
    %4 = tensor.expand_shape %3 [[0, 1, 2], [3]] output_shape [1, 1, 5, 3] : tensor<5x3xf32> into tensor<1x1x5x3xf32>
    %5 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%4, %fc : tensor<1x1x5x3xf32>, tensor<2x1x1x1xf32>) outs(%fc_bias : tensor<1x2x5x3xf32>) -> tensor<1x2x5x3xf32>
    secret.yield %5 : tensor<1x2x5x3xf32>
  } -> !secret.secret<tensor<1x2x5x3xf32>>
  return %0 : !secret.secret<tensor<1x2x5x3xf32>>
}

// Two secret operands with coprime dims select the ct-ct bicyclic kernel.
// CHECK: @strided_conv_bicyclic_matmul_conv
// CHECK: linalg.matmul
// CHECK-SAME: heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 5, 3>}
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-SAME: heir.kernel_info = {gap_factor = 1 : i64, input_shape = array<i64: 1, 1, 5, 3>, result_shape = array<i64: 1, 2, 5, 3>}
func.func @strided_conv_bicyclic_matmul_conv(%arg0: !secret.secret<tensor<1x1x10x10xf32>>) -> !secret.secret<tensor<1x2x5x3xf32>> {
  %filter = arith.constant dense<2.500000e-01> : tensor<4x1x2x2xf32>
  %bias = arith.constant dense<1.000000e+00> : tensor<1x4x5x5xf32>
  %init = arith.constant dense<0.000000e+00> : tensor<5x3xf32>
  %fc = arith.constant dense<5.000000e-01> : tensor<2x1x1x1xf32>
  %fc_bias = arith.constant dense<1.000000e+00> : tensor<1x2x5x3xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x10x10xf32>>) {
  ^body(%input0: tensor<1x1x10x10xf32>):
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%input0, %filter : tensor<1x1x10x10xf32>, tensor<4x1x2x2xf32>) outs(%bias : tensor<1x4x5x5xf32>) -> tensor<1x4x5x5xf32>
    %2 = tensor.extract_slice %1[0, 0, 0, 0] [1, 1, 5, 4] [1, 1, 1, 1] : tensor<1x4x5x5xf32> to tensor<5x4xf32>
    %3 = tensor.extract_slice %1[0, 1, 0, 0] [1, 1, 4, 3] [1, 1, 1, 1] : tensor<1x4x5x5xf32> to tensor<4x3xf32>
    %4 = linalg.matmul ins(%2, %3 : tensor<5x4xf32>, tensor<4x3xf32>) outs(%init : tensor<5x3xf32>) -> tensor<5x3xf32>
    %5 = tensor.expand_shape %4 [[0, 1, 2], [3]] output_shape [1, 1, 5, 3] : tensor<5x3xf32> into tensor<1x1x5x3xf32>
    %6 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%5, %fc : tensor<1x1x5x3xf32>, tensor<2x1x1x1xf32>) outs(%fc_bias : tensor<1x2x5x3xf32>) -> tensor<1x2x5x3xf32>
    secret.yield %6 : tensor<1x2x5x3xf32>
  } -> !secret.secret<tensor<1x2x5x3xf32>>
  return %0 : !secret.secret<tensor<1x2x5x3xf32>>
}

// The row-major conv_2d kernel. Without gap factor 1 on its result, the next
// conv reads the row-major data as gap-2 data.
// CHECK: @strided_conv_conv_2d_conv
// CHECK: linalg.conv_2d {
// CHECK-SAME: heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 3, 3>}
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-SAME: heir.kernel_info = {gap_factor = 1 : i64, input_shape = array<i64: 1, 1, 3, 3>, result_shape = array<i64: 1, 2, 3, 3>}
func.func @strided_conv_conv_2d_conv(%arg0: !secret.secret<tensor<1x1x8x8xf32>>) -> !secret.secret<tensor<1x2x3x3xf32>> {
  %filter = arith.constant dense<2.500000e-01> : tensor<4x1x2x2xf32>
  %bias = arith.constant dense<1.000000e+00> : tensor<1x4x4x4xf32>
  %filter2 = arith.constant dense<6.250000e-02> : tensor<2x2xf32>
  %init = arith.constant dense<0.000000e+00> : tensor<3x3xf32>
  %fc = arith.constant dense<5.000000e-01> : tensor<2x1x1x1xf32>
  %fc_bias = arith.constant dense<1.000000e+00> : tensor<1x2x3x3xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x8x8xf32>>) {
  ^body(%input0: tensor<1x1x8x8xf32>):
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%input0, %filter : tensor<1x1x8x8xf32>, tensor<4x1x2x2xf32>) outs(%bias : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    %2 = tensor.extract_slice %1[0, 0, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : tensor<1x4x4x4xf32> to tensor<4x4xf32>
    %3 = linalg.conv_2d ins(%2, %filter2 : tensor<4x4xf32>, tensor<2x2xf32>) outs(%init : tensor<3x3xf32>) -> tensor<3x3xf32>
    %4 = tensor.expand_shape %3 [[0, 1, 2], [3]] output_shape [1, 1, 3, 3] : tensor<3x3xf32> into tensor<1x1x3x3xf32>
    %5 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%4, %fc : tensor<1x1x3x3xf32>, tensor<2x1x1x1xf32>) outs(%fc_bias : tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32>
    secret.yield %5 : tensor<1x2x3x3xf32>
  } -> !secret.secret<tensor<1x2x3x3xf32>>
  return %0 : !secret.secret<tensor<1x2x3x3xf32>>
}

// The row-major conv_1d kernel.
// CHECK: @strided_conv_conv_1d
// CHECK: linalg.conv_1d {
// CHECK-SAME: heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 3>}
func.func @strided_conv_conv_1d(%arg0: !secret.secret<tensor<1x1x8x8xf32>>) -> !secret.secret<tensor<3xf32>> {
  %filter = arith.constant dense<2.500000e-01> : tensor<4x1x2x2xf32>
  %bias = arith.constant dense<1.000000e+00> : tensor<1x4x4x4xf32>
  %filter2 = arith.constant dense<5.000000e-01> : tensor<2xf32>
  %init = arith.constant dense<0.000000e+00> : tensor<3xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x8x8xf32>>) {
  ^body(%input0: tensor<1x1x8x8xf32>):
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%input0, %filter : tensor<1x1x8x8xf32>, tensor<4x1x2x2xf32>) outs(%bias : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    %2 = tensor.extract_slice %1[0, 0, 0, 0] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<1x4x4x4xf32> to tensor<4xf32>
    %3 = linalg.conv_1d ins(%2, %filter2 : tensor<4xf32>, tensor<2xf32>) outs(%init : tensor<3xf32>) -> tensor<3xf32>
    secret.yield %3 : tensor<3xf32>
  } -> !secret.secret<tensor<3xf32>>
  return %0 : !secret.secret<tensor<3xf32>>
}

// The dot kernel gives a scalar in the default layout.
// CHECK: @strided_conv_dot
// CHECK: linalg.dot
// CHECK-SAME: heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64>}
func.func @strided_conv_dot(%arg0: !secret.secret<tensor<1x1x8x8xf32>>) -> !secret.secret<tensor<f32>> {
  %filter = arith.constant dense<2.500000e-01> : tensor<4x1x2x2xf32>
  %bias = arith.constant dense<1.000000e+00> : tensor<1x4x4x4xf32>
  %vector = arith.constant dense<5.000000e-01> : tensor<4xf32>
  %init = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x8x8xf32>>) {
  ^body(%input0: tensor<1x1x8x8xf32>):
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%input0, %filter : tensor<1x1x8x8xf32>, tensor<4x1x2x2xf32>) outs(%bias : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    %2 = tensor.extract_slice %1[0, 0, 0, 0] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<1x4x4x4xf32> to tensor<4xf32>
    %3 = linalg.dot ins(%2, %vector : tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<f32>) -> tensor<f32>
    secret.yield %3 : tensor<f32>
  } -> !secret.secret<tensor<f32>>
  return %0 : !secret.secret<tensor<f32>>
}

// A broadcast gives a new cyclic or row-major layout.
// CHECK: @strided_conv_broadcast
// CHECK: linalg.broadcast
// CHECK-SAME: heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 4, 4>}
func.func @strided_conv_broadcast(%arg0: !secret.secret<tensor<1x1x8x8xf32>>) -> !secret.secret<tensor<4x4xf32>> {
  %filter = arith.constant dense<2.500000e-01> : tensor<4x1x2x2xf32>
  %bias = arith.constant dense<1.000000e+00> : tensor<1x4x4x4xf32>
  %init = arith.constant dense<0.000000e+00> : tensor<4x4xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x8x8xf32>>) {
  ^body(%input0: tensor<1x1x8x8xf32>):
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%input0, %filter : tensor<1x1x8x8xf32>, tensor<4x1x2x2xf32>) outs(%bias : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    %2 = tensor.extract_slice %1[0, 0, 0, 0] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<1x4x4x4xf32> to tensor<4xf32>
    %3 = linalg.broadcast ins(%2 : tensor<4xf32>) outs(%init : tensor<4x4xf32>) dimensions = [0]
    secret.yield %3 : tensor<4x4xf32>
  } -> !secret.secret<tensor<4x4xf32>>
  return %0 : !secret.secret<tensor<4x4xf32>>
}

// The tricyclic batch_matmul kernel sets no kernel info itself, so this case
// tests the ungapped default. Coprime dims select the tricyclic kernel.
// CHECK: @strided_conv_batch_matmul
// CHECK: linalg.batch_matmul
// CHECK-SAME: heir.kernel_info = {gap_factor = 1 : i64, result_shape = array<i64: 1, 5, 3>}
func.func @strided_conv_batch_matmul(%arg0: !secret.secret<tensor<1x1x10x10xf32>>) -> !secret.secret<tensor<1x5x3xf32>> {
  %filter = arith.constant dense<2.500000e-01> : tensor<4x1x2x2xf32>
  %bias = arith.constant dense<1.000000e+00> : tensor<1x4x5x5xf32>
  %matrix = arith.constant dense<6.250000e-02> : tensor<1x4x3xf32>
  %init = arith.constant dense<0.000000e+00> : tensor<1x5x3xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x10x10xf32>>) {
  ^body(%input0: tensor<1x1x10x10xf32>):
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%input0, %filter : tensor<1x1x10x10xf32>, tensor<4x1x2x2xf32>) outs(%bias : tensor<1x4x5x5xf32>) -> tensor<1x4x5x5xf32>
    %2 = tensor.extract_slice %1[0, 0, 0, 0] [1, 1, 5, 4] [1, 1, 1, 1] : tensor<1x4x5x5xf32> to tensor<1x5x4xf32>
    %3 = linalg.batch_matmul ins(%2, %matrix : tensor<1x5x4xf32>, tensor<1x4x3xf32>) outs(%init : tensor<1x5x3xf32>) -> tensor<1x5x3xf32>
    secret.yield %3 : tensor<1x5x3xf32>
  } -> !secret.secret<tensor<1x5x3xf32>>
  return %0 : !secret.secret<tensor<1x5x3xf32>>
}

// An elementwise op keeps its operand's gapped layout, so it keeps the gap.
// CHECK: @strided_conv_elementwise
// CHECK: arith.mulf
// CHECK-SAME: heir.kernel_info = {gap_factor = 2 : i64, input_shape = array<i64: 1, 1, 8, 8>, result_shape = array<i64: 4, 4>}
func.func @strided_conv_elementwise(%arg0: !secret.secret<tensor<1x1x8x8xf32>>) -> !secret.secret<tensor<4x4xf32>> {
  %filter = arith.constant dense<2.500000e-01> : tensor<4x1x2x2xf32>
  %bias = arith.constant dense<1.000000e+00> : tensor<1x4x4x4xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x8x8xf32>>) {
  ^body(%input0: tensor<1x1x8x8xf32>):
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%input0, %filter : tensor<1x1x8x8xf32>, tensor<4x1x2x2xf32>) outs(%bias : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    %2 = tensor.extract_slice %1[0, 0, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : tensor<1x4x4x4xf32> to tensor<4x4xf32>
    %3 = arith.mulf %2, %2 : tensor<4x4xf32>
    secret.yield %3 : tensor<4x4xf32>
  } -> !secret.secret<tensor<4x4xf32>>
  return %0 : !secret.secret<tensor<4x4xf32>>
}
