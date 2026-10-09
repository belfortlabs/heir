// RUN: heir-opt --split-input-file --layout-propagation %s | FileCheck %s

// The matvec kernel holds a whole diagonal of the 4x1500 matrix in each
// ciphertext, so the default 1024 slots are raised to 2048.

// CHECK: 0 <= slot <= 2047
// CHECK: module attributes {scheme.layout_slot_count = 2048 : i64}
// CHECK: @matvec
func.func @matvec(%arg0: !secret.secret<tensor<1500xf32>>) -> !secret.secret<tensor<4xf32>> {
  %cst = arith.constant dense<0.0> : tensor<4xf32>
  %matrix = arith.constant dense<1.0> : tensor<4x1500xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1500xf32>>) {
  ^body(%input0: tensor<1500xf32>):
    %1 = linalg.matvec ins(%matrix, %input0 : tensor<4x1500xf32>, tensor<1500xf32>) outs(%cst : tensor<4xf32>) -> tensor<4xf32>
    secret.yield %1 : tensor<4xf32>
  } -> !secret.secret<tensor<4xf32>>
  return %0 : !secret.secret<tensor<4xf32>>
}

// -----

// The strided conv's expanded filter matrix has 8 * 16 * 16 = 2048 rows and
// 3 * 32 * 32 = 3072 columns, so it needs 4096 slots.

// CHECK: 0 <= slot <= 4095
// CHECK: module attributes {scheme.layout_slot_count = 4096 : i64}
// CHECK: @conv2d_nchw_strided
func.func @conv2d_nchw_strided(%arg0: !secret.secret<tensor<1x3x32x32xf32>>) -> !secret.secret<tensor<1x8x16x16xf32>> {
  %cst = arith.constant dense<0.0> : tensor<1x8x16x16xf32>
  %filter = arith.constant dense<0.25> : tensor<8x3x2x2xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x3x32x32xf32>>) {
  ^body(%input0: tensor<1x3x32x32xf32>):
    %1 = linalg.conv_2d_nchw_fchw
      { dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
      ins(%input0, %filter : tensor<1x3x32x32xf32>, tensor<8x3x2x2xf32>)
      outs(%cst : tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
    secret.yield %1 : tensor<1x8x16x16xf32>
  } -> !secret.secret<tensor<1x8x16x16xf32>>
  return %0 : !secret.secret<tensor<1x8x16x16xf32>>
}

// -----

// A kernel that fits in min-slot-count leaves the slot count alone.

// CHECK-NOT: scheme.layout_slot_count
// CHECK: @small_matvec
func.func @small_matvec(%arg0: !secret.secret<tensor<16xf32>>) -> !secret.secret<tensor<4xf32>> {
  %cst = arith.constant dense<0.0> : tensor<4xf32>
  %matrix = arith.constant dense<1.0> : tensor<4x16xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<16xf32>>) {
  ^body(%input0: tensor<16xf32>):
    %1 = linalg.matvec ins(%matrix, %input0 : tensor<4x16xf32>, tensor<16xf32>) outs(%cst : tensor<4xf32>) -> tensor<4xf32>
    secret.yield %1 : tensor<4xf32>
  } -> !secret.secret<tensor<4xf32>>
  return %0 : !secret.secret<tensor<4xf32>>
}
