// RUN: heir-opt --layout-optimization --canonicalize %s | FileCheck %s

// A matvec input conversion must not fold into the weights when the permuted
// vector does not fill the slots below its size, as after a strided
// convolution. The folded matrix only has columns below the vector size, and a
// squat-diagonal kernel with a period smaller than the ciphertext reads the
// row-major replica the permuted vector lacks. The conversion then moves into
// the argument's layout and the matrix keeps its own.

#gapped = #tensor_ext.layout<"{ [i] -> [ct, slot] : ct = 0 and slot = 2i and 0 <= i <= 3 and 0 <= slot <= 7 }">
#row_major_replicated = #tensor_ext.layout<"{ [i] -> [ct, slot] : ct = 0 and (slot - i) mod 4 = 0 and 0 <= i <= 3 and 0 <= slot <= 7 }">
#matrix_2x4 = #tensor_ext.layout<"{ [row, col] -> [ct, slot] : (row - col + ct) mod 2 = 0 and (-col + ct + slot) mod 4 = 0 and 0 <= row <= 1 and 0 <= col <= 3 and 0 <= ct <= 1 and 0 <= slot <= 7 }">
#shifted = #tensor_ext.layout<"{ [i] -> [ct, slot] : ct = 0 and slot = i + 2 and 0 <= i <= 5 and 0 <= slot <= 7 }">
#row_major_6 = #tensor_ext.layout<"{ [i] -> [ct, slot] : ct = 0 and (slot - i) mod 8 = 0 and 0 <= i <= 5 and 0 <= slot <= 7 }">
#matrix_2x6 = #tensor_ext.layout<"{ [row, col] -> [ct, slot] : (row - col + ct) mod 2 = 0 and (-col + ct + slot) mod 8 = 0 and 0 <= row <= 1 and 0 <= col <= 5 and 0 <= ct <= 1 and 0 <= slot <= 7 }">
#result = #tensor_ext.layout<"{ [i] -> [ct, slot] : ct = 0 and (slot - i) mod 2 = 0 and 0 <= i <= 1 and 0 <= slot <= 7 }">

// CHECK-DAG: #[[ROW_MAJOR_REPLICATED:layout[0-9]*]] = #tensor_ext.layout<"{ [i] -> [ct, slot] : ct = 0 and (slot - i) mod 4 = 0 and 0 <= i <= 3 and 0 <= slot <= 7 }">
// CHECK-DAG: #[[MATRIX_2X4:layout[0-9]*]] = #tensor_ext.layout<"{ [row, col] -> [ct, slot] : (row - col + ct) mod 2 = 0 and (-col + ct + slot) mod 4 = 0 and 0 <= row <= 1 and 0 <= col <= 3 and 0 <= ct <= 1 and 0 <= slot <= 7 }">
// CHECK-DAG: #[[ROW_MAJOR_6:layout[0-9]*]] = #tensor_ext.layout<"{ [i] -> [ct, slot] : ct = 0 and (slot - i) mod 8 = 0 and 0 <= i <= 5 and 0 <= slot <= 7 }">
// CHECK-DAG: #[[MATRIX_2X6:layout[0-9]*]] = #tensor_ext.layout<"{ [row, col] -> [ct, slot] : (row - col + ct) mod 2 = 0 and (-col + ct + slot) mod 8 = 0 and 0 <= row <= 1 and 0 <= col <= 5 and 0 <= ct <= 1 and 0 <= slot <= 7 }">

// The kernel's period (4) is smaller than the ciphertext (8 slots), and the
// gapped vector reaches slot 6, past the matrix's 4 columns.

// CHECK: func.func @gapped_input_to_replicated_row_major(
// CHECK-SAME: tensor_ext.layout = #[[ROW_MAJOR_REPLICATED]]
// CHECK: tensor_ext.assign_layout %{{.*}} {layout = #[[MATRIX_2X4]],
// CHECK: linalg.matvec
func.func @gapped_input_to_replicated_row_major(
    %weights: tensor<2x4xf32>,
    %input: !secret.secret<tensor<4xf32>> {tensor_ext.layout = #gapped})
    -> (!secret.secret<tensor<2xf32>> {tensor_ext.layout = #result}) {
  %empty = tensor.empty() : tensor<2xf32>
  %init = tensor_ext.assign_layout %empty {layout = #result, tensor_ext.layout = #result} : tensor<2xf32>
  %matrix = tensor_ext.assign_layout %weights {layout = #matrix_2x4, tensor_ext.layout = #matrix_2x4} : tensor<2x4xf32>
  %0 = secret.generic(
      %input: !secret.secret<tensor<4xf32>> {tensor_ext.layout = #gapped}) {
  ^body(%arg0: tensor<4xf32>):
    %converted = tensor_ext.convert_layout %arg0 {from_layout = #gapped, tensor_ext.layout = #row_major_replicated, to_layout = #row_major_replicated} : tensor<4xf32>
    %result = linalg.matvec {secret.kernel = #secret.kernel<name="MatvecDiagonal", force=false>, tensor_ext.layout = #result}
        ins(%matrix, %converted : tensor<2x4xf32>, tensor<4xf32>)
        outs(%init : tensor<2xf32>) -> tensor<2xf32>
    secret.yield %result : tensor<2xf32>
  } -> !secret.secret<tensor<2xf32>> {tensor_ext.layout = #result}
  return %0 : !secret.secret<tensor<2xf32>>
}

// The kernel's period is the whole ciphertext, but the shifted vector fills
// slots 2 to 7, past the matrix's 6 columns.

// CHECK: func.func @shifted_input_past_the_vector_size(
// CHECK-SAME: tensor_ext.layout = #[[ROW_MAJOR_6]]
// CHECK: tensor_ext.assign_layout %{{.*}} {layout = #[[MATRIX_2X6]],
// CHECK: linalg.matvec
func.func @shifted_input_past_the_vector_size(
    %weights: tensor<2x6xf32>,
    %input: !secret.secret<tensor<6xf32>> {tensor_ext.layout = #shifted})
    -> (!secret.secret<tensor<2xf32>> {tensor_ext.layout = #result}) {
  %empty = tensor.empty() : tensor<2xf32>
  %init = tensor_ext.assign_layout %empty {layout = #result, tensor_ext.layout = #result} : tensor<2xf32>
  %matrix = tensor_ext.assign_layout %weights {layout = #matrix_2x6, tensor_ext.layout = #matrix_2x6} : tensor<2x6xf32>
  %0 = secret.generic(
      %input: !secret.secret<tensor<6xf32>> {tensor_ext.layout = #shifted}) {
  ^body(%arg0: tensor<6xf32>):
    %converted = tensor_ext.convert_layout %arg0 {from_layout = #shifted, tensor_ext.layout = #row_major_6, to_layout = #row_major_6} : tensor<6xf32>
    %result = linalg.matvec {secret.kernel = #secret.kernel<name="MatvecDiagonal", force=false>, tensor_ext.layout = #result}
        ins(%matrix, %converted : tensor<2x6xf32>, tensor<6xf32>)
        outs(%init : tensor<2xf32>) -> tensor<2xf32>
    secret.yield %result : tensor<2xf32>
  } -> !secret.secret<tensor<2xf32>> {tensor_ext.layout = #result}
  return %0 : !secret.secret<tensor<2xf32>>
}
