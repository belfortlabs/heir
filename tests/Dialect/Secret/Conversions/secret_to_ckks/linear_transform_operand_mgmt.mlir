// RUN: heir-opt --secret-to-ckks %s | FileCheck %s

// The generic's operand mgmt attr is one level above its result's. The converted
// linear_transform must keep the result's level, or the return needs an
// unresolved materialization.

// CHECK: func.func @lintrans_operand_mgmt
// CHECK-SAME: -> tensor<1x!ct_L1>
// CHECK: kernel.linear_transform
// CHECK-SAME: -> tensor<1x!ct_L1>
// CHECK-NEXT: return {{.*}} : tensor<1x!ct_L1>
module attributes {
  ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797017456641, 35184371138561, 35184372121601], P = [1152921504607338497, 1152921504608747521], logDefaultScale = 45, encryptionTechnique = extended>,
  scheme.ckks,
  scheme.requested_slot_count = 1024 : i64
} {
  func.func @lintrans_operand_mgmt(
      %arg0: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 45>})
      -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>}) {
    %cst = arith.constant dense<1.000000e+00> : tensor<2x1024xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 45>}) {
    ^body(%input0: tensor<1x1024xf32>):
      %1 = kernel.linear_transform %input0, %cst {diagonal_indices = array<i64: 0, 1>, mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>} : tensor<1x1024xf32>, tensor<2x1024xf32> -> tensor<1x1024xf32>
      secret.yield %1 : tensor<1x1024xf32>
    } -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>})
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}
