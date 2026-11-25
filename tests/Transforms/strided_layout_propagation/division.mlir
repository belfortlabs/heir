// RUN: heir-opt --strided-layout-propagation=min-slot-count=1024 %s | FileCheck %s

// A division by a cleartext tensor becomes a multiplication by its
// reciprocal, which is computed in cleartext. The packed reciprocal is zero
// in the slots that hold no element, where a packed divisor would divide by
// zero.

// CHECK: func.func @divide
// CHECK-SAME: %[[d:[^:]*]]: tensor<8xf32>
// CHECK: %[[one:.*]] = arith.constant dense<1.000000e+00> : tensor<8xf32>
// CHECK: %[[reciprocal:.*]] = arith.divf %[[one]], %[[d]]
// CHECK: secret.generic
// CHECK: %[[packed:.*]] = tensor_ext.assign_layout %[[reciprocal]]
// CHECK: arith.mulf %{{.*}}, %[[packed]]
// CHECK-NOT: arith.divf
// CHECK: return
func.func @divide(%x: !secret.secret<tensor<8xf32>>, %d: tensor<8xf32>) -> !secret.secret<tensor<8xf32>> {
  %0 = secret.generic(%x: !secret.secret<tensor<8xf32>>) {
  ^body(%x0: tensor<8xf32>):
    %q = arith.divf %x0, %d : tensor<8xf32>
    secret.yield %q : tensor<8xf32>
  } -> !secret.secret<tensor<8xf32>>
  return %0 : !secret.secret<tensor<8xf32>>
}
