// RUN: heir-opt --layout-propagation=min-slot-count=1024 %s | FileCheck %s

// A packing already on the input -- left by an earlier run of this pass, or
// preserved across an op clone -- must be replaced by the one this run decides.
// Left in place it would make ConvertToCiphertextSemantics size the Toeplitz
// matrix against an operand this ciphertext never holds.
//
// One attribute carries every field, so this run cannot leave a stale field
// beside a fresh one: it either writes the whole packing or none of it.

// CHECK-DAG: #[[$PACK:.*]] = #tensor_ext.conv_packing<matrixDataType = tensor<1x4x4x4xf32>, padding = 0, interchangeRows = false, absorbedMatrixWidth = 0>
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-SAME: tensor_ext.conv_packing = #[[$PACK]]

#stale = #tensor_ext.conv_packing<matrixDataType = tensor<1x4x2x2xf32>, padding = 1, interchangeRows = true, absorbedMatrixWidth = 1024>

func.func @stale_conv_packing(%arg0: !secret.secret<tensor<1x4x4x4xf32>>) -> !secret.secret<tensor<1x4x4x4xf32>> {
  %out = arith.constant dense<0.000000e+00> : tensor<1x4x4x4xf32>
  %filter = arith.constant dense<3.000000e+00> : tensor<4x4x1x1xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x4x4x4xf32>>) {
  ^body(%input0: tensor<1x4x4x4xf32>):
    // No tensor.pad to fold, but the op arrives already carrying a packing.
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tensor_ext.conv_packing = #stale} ins(%input0, %filter : tensor<1x4x4x4xf32>, tensor<4x4x1x1xf32>) outs(%out : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    secret.yield %1 : tensor<1x4x4x4xf32>
  } -> !secret.secret<tensor<1x4x4x4xf32>>
  return %0 : !secret.secret<tensor<1x4x4x4xf32>>
}
