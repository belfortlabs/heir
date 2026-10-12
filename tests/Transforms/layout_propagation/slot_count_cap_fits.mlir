// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" --layout-propagation %s | FileCheck %s

// A diagonal of the 4x1500 matrix needs 2048 slots, which the 32768 slots of
// the backend's logN = 16 ring hold, so the slot count is raised without error.

// CHECK: 0 <= slot <= 2047
// CHECK: module attributes
// CHECK-SAME: scheme.layout_slot_count = 2048 : i64
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
