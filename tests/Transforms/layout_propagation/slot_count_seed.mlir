// RUN: heir-opt --split-input-file --layout-propagation=min-slot-count=4096 %s | FileCheck %s

// The matvec needs 2048 slots, fewer than min-slot-count, so the option's 4096
// slots are kept and recorded.

// CHECK: 0 <= slot <= 4095
// CHECK: module attributes {scheme.layout_slot_count = 4096 : i64}
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

// A slot count already recorded on the module above min-slot-count is kept.

// CHECK: 0 <= slot <= 8191
// CHECK: module attributes {scheme.layout_slot_count = 8192 : i64}
// CHECK: @recorded_matvec
module attributes {scheme.layout_slot_count = 8192 : i64} {
  func.func @recorded_matvec(%arg0: !secret.secret<tensor<1500xf32>>) -> !secret.secret<tensor<4xf32>> {
    %cst = arith.constant dense<0.0> : tensor<4xf32>
    %matrix = arith.constant dense<1.0> : tensor<4x1500xf32>
    %0 = secret.generic(%arg0 : !secret.secret<tensor<1500xf32>>) {
    ^body(%input0: tensor<1500xf32>):
      %1 = linalg.matvec ins(%matrix, %input0 : tensor<4x1500xf32>, tensor<1500xf32>) outs(%cst : tensor<4xf32>) -> tensor<4xf32>
      secret.yield %1 : tensor<4xf32>
    } -> !secret.secret<tensor<4xf32>>
    return %0 : !secret.secret<tensor<4xf32>>
  }
}
