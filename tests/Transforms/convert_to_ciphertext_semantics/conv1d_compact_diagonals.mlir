// RUN: heir-opt --layout-propagation=min-slot-count=64 --convert-to-ciphertext-semantics=min-slot-count=64 %s | FileCheck %s

// The nonzero diagonals of a constant filter are gathered during conversion and
// the full packed matrix is dropped.

// A 1-tap filter over 4 channels of 8 positions fills only the diagonals at
// multiples of 8; the transform receives those 4 of the 32 rows.
// CHECK: func.func @conv1d_resource_filter
// CHECK: %[[DIAGS:.*]] = arith.constant dense_resource<conv_filter_packed_gathered> : tensor<4x64xf32>
// CHECK: tensor_ext.rotate_and_reduce %{{.*}}, %[[DIAGS]]
// CHECK-SAME: steps = 32
// CHECK-SAME: tensor_ext.diagonal_indices = array<i32: 0, 8, 16, 24>
// CHECK-NOT: conv_filter_packed:
// Diagonal 0 holds the filter's main diagonal W[c][c] = 1, 6, 11, 16, each
// repeated over the 8 positions, and the 32 slots again in the upper half.
// CHECK: conv_filter_packed_gathered: "0x040000000000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000C040

module attributes {
  backend.openfhe,
  backend.config_override = {has_kernel_linear_transform = true}
} {
func.func @conv1d_resource_filter(%arg0: !secret.secret<tensor<1x4x8xf32>>) -> !secret.secret<tensor<1x4x8xf32>> {
  %out = arith.constant dense<0.000000e+00> : tensor<1x4x8xf32>
  %filter = arith.constant dense_resource<conv_filter> : tensor<4x4x1xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x4x8xf32>>) {
  ^body(%input0: tensor<1x4x8xf32>):
    %1 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%input0, %filter : tensor<1x4x8xf32>, tensor<4x4x1xf32>) outs(%out : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    secret.yield %1 : tensor<1x4x8xf32>
  } -> !secret.secret<tensor<1x4x8xf32>>
  return %0 : !secret.secret<tensor<1x4x8xf32>>
}
}

{-#
  dialect_resources: {
    builtin: {
      conv_filter: "0x040000000000803F0000004000004040000080400000A0400000C0400000E040000000410000104100002041000030410000404100005041000060410000704100008041"
    }
  }
#-}
