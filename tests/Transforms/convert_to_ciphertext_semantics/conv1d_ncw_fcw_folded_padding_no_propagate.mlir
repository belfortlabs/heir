// RUN: heir-opt --layout-propagation=min-slot-count=1024 %s | FileCheck %s

// A packing describes the one op whose filter matrix was built against an
// unpadded operand. It must not travel along the value chain to %2's conv,
// whose operand was never padded: that conv records its own packing, with a
// padding of 0. It absorbs the gapped packing the strided conv left behind,
// which is why its own matrix is built at the ciphertext width.

// CHECK-DAG: #[[$PACK1:.*]] = #tensor_ext.conv_packing<matrixDataType = tensor<1x8x6xf32>, padding = 2, interchangeRows = true, absorbedMatrixWidth = 0>
// CHECK-DAG: #[[$PACK2:.*]] = #tensor_ext.conv_packing<matrixDataType = tensor<1x8x4xf32>, padding = 0, interchangeRows = true, absorbedMatrixWidth = 1024>
// CHECK: linalg.conv_1d_ncw_fcw
// CHECK-SAME: tensor_ext.conv_packing = #[[$PACK1]]
// CHECK: linalg.conv_1d_ncw_fcw
// CHECK-SAME: tensor_ext.conv_packing = #[[$PACK2]]

func.func @padded_then_unpadded(%arg0: !secret.secret<tensor<1x8x6xf32>>) -> !secret.secret<tensor<1x8x4xf32>> {
  %out = arith.constant dense<0.000000e+00> : tensor<1x8x4xf32>
  %filter = arith.constant dense<2.000000e+00> : tensor<8x8x3xf32>
  %filter1 = arith.constant dense<3.000000e+00> : tensor<8x8x1xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x8x6xf32>>) {
  ^body(%input0: tensor<1x8x6xf32>):
    %zero = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %input0 low[0, 0, 2] high[0, 0, 2] {
    ^bb0(%i: index, %j: index, %k: index):
      tensor.yield %zero : f32
    } : tensor<1x8x6xf32> to tensor<1x8x10xf32>
    %1 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>} ins(%padded, %filter : tensor<1x8x10xf32>, tensor<8x8x3xf32>) outs(%out : tensor<1x8x4xf32>) -> tensor<1x8x4xf32>
    // Unpadded, stride 1, single tap: consumes the previous conv's result.
    %2 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%1, %filter1 : tensor<1x8x4xf32>, tensor<8x8x1xf32>) outs(%out : tensor<1x8x4xf32>) -> tensor<1x8x4xf32>
    secret.yield %2 : tensor<1x8x4xf32>
  } -> !secret.secret<tensor<1x8x4xf32>>
  return %0 : !secret.secret<tensor<1x8x4xf32>>
}
