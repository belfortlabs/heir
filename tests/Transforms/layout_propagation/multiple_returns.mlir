// RUN: heir-opt --layout-propagation --fold-convert-layout-into-assign-layout %s | FileCheck %s

// Don't insert layout rectification for a func with multiple return operands

// CHECK: @add
// CHECK-SAME: %[[arg0:.*]]: !secret.secret<tensor<16xf32>>
// CHECK-SAME: %[[arg1:.*]]: !secret.secret<tensor<5xf32>>
func.func @add(%arg0: !secret.secret<tensor<16xf32>>, %arg1: !secret.secret<tensor<5xf32>>) -> (!secret.secret<tensor<16xf32>>, !secret.secret<tensor<5xf32>>) {
  // CHECK: %[[cst:.*]] = arith.constant dense<1.0

  // Assign a layout to the cleartext operand
  // CHECK: tensor_ext.assign_layout %[[cst]]
  %cst = arith.constant dense<1.000000e+00> : tensor<16xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<16xf32>>) {
  ^body(%input0: tensor<16xf32>):
    // CHECK: arith.addf
    %1 = arith.addf %input0, %cst : tensor<16xf32>
    secret.yield %1 : tensor<16xf32>
    // CHECK: secret.yield
    // CHECK-NOT: tensor_ext.convert_layout %[[arg1]]
    // CHECK: return
  } -> !secret.secret<tensor<16xf32>>
  return %0, %arg1 : !secret.secret<tensor<16xf32>>, !secret.secret<tensor<5xf32>>
}
