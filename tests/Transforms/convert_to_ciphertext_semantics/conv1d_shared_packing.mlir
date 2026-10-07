// RUN: heir-opt --layout-propagation=min-slot-count=64 --cse --convert-to-ciphertext-semantics=min-slot-count=64 %s | FileCheck %s

// With --cse both convolutions consume one assign_layout. The second reuses the
// diagonals gathered for the first, whose full packed matrix is freed.

// CHECK: func.func @shared_filter
// CHECK: %[[DIAGS:.*]] = arith.constant dense_resource<conv_filter_packed_gathered> : tensor<4x64xf32>
// CHECK: tensor_ext.rotate_and_reduce %{{.*}}, %[[DIAGS]]
// CHECK: tensor_ext.rotate_and_reduce %{{.*}}, %[[DIAGS]]
// CHECK: dialect_resources
// CHECK-NEXT: builtin
// CHECK-NEXT: conv_filter_packed_gathered: "0x
// CHECK-NEXT: }

module attributes {
  backend.openfhe,
  backend.config_override = {has_kernel_linear_transform = true}
} {
func.func @shared_filter(%arg0: !secret.secret<tensor<1x4x8xf32>>, %arg1: !secret.secret<tensor<1x4x8xf32>>) -> (!secret.secret<tensor<1x4x8xf32>>, !secret.secret<tensor<1x4x8xf32>>) {
  %out = arith.constant dense<0.000000e+00> : tensor<1x4x8xf32>
  %filter = arith.constant dense_resource<conv_filter> : tensor<4x4x1xf32>
  %0:2 = secret.generic(%arg0 : !secret.secret<tensor<1x4x8xf32>>, %arg1 : !secret.secret<tensor<1x4x8xf32>>) {
  ^body(%input0: tensor<1x4x8xf32>, %input1: tensor<1x4x8xf32>):
    %1 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%input0, %filter : tensor<1x4x8xf32>, tensor<4x4x1xf32>) outs(%out : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %2 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%input1, %filter : tensor<1x4x8xf32>, tensor<4x4x1xf32>) outs(%out : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    secret.yield %1, %2 : tensor<1x4x8xf32>, tensor<1x4x8xf32>
  } -> (!secret.secret<tensor<1x4x8xf32>>, !secret.secret<tensor<1x4x8xf32>>)
  return %0#0, %0#1 : !secret.secret<tensor<1x4x8xf32>>, !secret.secret<tensor<1x4x8xf32>>
}
}

{-#
  dialect_resources: {
    builtin: {
      conv_filter: "0x040000000000803F0000004000004040000080400000A0400000C0400000E040000000410000104100002041000030410000404100005041000060410000704100008041"
    }
  }
#-}
