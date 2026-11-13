// RUN: heir-opt --layout-propagation %s | FileCheck %s

// The first strided conv fits in 4096 slots, but it leaves its output
// gapped, and the second conv reads that gapped input and needs more. Layout
// propagation restarts with the larger count, so the first conv is packed for
// it too.

// CHECK-NOT: slot <= 4095
// CHECK: (-1024i1 - 32i2 - i3 + slot) mod 4096 = 0 and 0 <= i1 <= 2 and 0 <= i2 <= 31 and 0 <= i3 <= 31 and 0 <= slot <= 8191
// CHECK-NOT: slot <= 4095
// CHECK: module attributes {scheme.layout_slot_count = 8192 : i64}
// CHECK: @two_strided_convs
func.func @two_strided_convs(%arg0: !secret.secret<tensor<1x3x32x32xf32>>) -> !secret.secret<tensor<1x128x8x8xf32>> {
  %cst = arith.constant dense<0.0> : tensor<1x8x16x16xf32>
  %cst2 = arith.constant dense<0.0> : tensor<1x128x8x8xf32>
  %filter = arith.constant dense<0.25> : tensor<8x3x2x2xf32>
  %filter2 = arith.constant dense<0.5> : tensor<128x8x2x2xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x3x32x32xf32>>) {
  ^body(%input0: tensor<1x3x32x32xf32>):
    %1 = linalg.conv_2d_nchw_fchw
      { dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
      ins(%input0, %filter : tensor<1x3x32x32xf32>, tensor<8x3x2x2xf32>)
      outs(%cst : tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %2 = linalg.conv_2d_nchw_fchw
      { dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
      ins(%1, %filter2 : tensor<1x8x16x16xf32>, tensor<128x8x2x2xf32>)
      outs(%cst2 : tensor<1x128x8x8xf32>) -> tensor<1x128x8x8xf32>
    secret.yield %2 : tensor<1x128x8x8xf32>
  } -> !secret.secret<tensor<1x128x8x8xf32>>
  return %0 : !secret.secret<tensor<1x128x8x8xf32>>
}
