// RUN: heir-opt --secret-to-ckks %s | FileCheck %s

// The generic around a linear transform carries the operand's mgmt attribute
// (level 2) next to the result's (level 1). The converted linear transform must
// keep the result's level, or the function return expects a level-2 ciphertext
// and the conversion leaves an unresolved materialization behind.

// CHECK: func.func @lintrans_operand_mgmt
// CHECK-SAME: tensor<1x!ct_L1>
// CHECK: kernel.linear_transform
// CHECK-SAME: -> tensor<1x!ct_L1>
// CHECK-NEXT: return {{.*}} : tensor<1x!ct_L1>
module attributes {
  backend.lattigo,
  ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797017456641, 35184371138561, 35184372121601], P = [1152921504607338497, 1152921504608747521], logDefaultScale = 45, encryptionTechnique = extended>,
  scheme.ckks,
  scheme.actual_slot_count = 8192 : i64,
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
