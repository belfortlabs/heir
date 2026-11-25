// RUN: heir-opt --convert-to-ciphertext-semantics=min-slot-count=16 %s | FileCheck %s

// Bootstrap placement runs after layout propagation, so the bootstrap op
// carries no layout of its own and takes the strided layout of its input.

#strided_layout = #tensor_ext.strided_layout<shape = [8], slots = 16, digits = [(0, 1, 0, 1)]>

// CHECK: func.func @square
// CHECK-SAME: !secret.secret<tensor<1x16xf32>>
// CHECK: ^body(%[[x:.*]]: tensor<1x16xf32>):
// CHECK: %[[mul:.*]] = arith.mulf %[[x]], %[[x]]
// CHECK: %[[boot:.*]] = mgmt.bootstrap %[[mul]]
// CHECK-SAME: tensor<1x16xf32>
// CHECK: %[[add:.*]] = arith.addf %[[boot]], %[[x]]
// CHECK: secret.yield %[[add]] : tensor<1x16xf32>
func.func @square(%arg0: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #strided_layout}) -> (!secret.secret<tensor<8xf32>> {tensor_ext.layout = #strided_layout}) {
  %0 = secret.generic(%arg0: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #strided_layout}) {
  ^body(%input0: tensor<8xf32>):
    %1 = arith.mulf %input0, %input0 {tensor_ext.layout = #strided_layout} : tensor<8xf32>
    %b = mgmt.bootstrap %1 : tensor<8xf32>
    %2 = arith.addf %b, %input0 {tensor_ext.layout = #strided_layout} : tensor<8xf32>
    secret.yield %2 : tensor<8xf32>
  } -> (!secret.secret<tensor<8xf32>> {tensor_ext.layout = #strided_layout})
  return %0 : !secret.secret<tensor<8xf32>>
}
