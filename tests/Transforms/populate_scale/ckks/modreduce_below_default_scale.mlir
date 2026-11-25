// RUN: heir-opt %s --populate-scale-ckks | FileCheck %s
// RUN: heir-opt %s --populate-scale-ckks --secret-distribute-generic --secret-to-ckks | FileCheck %s --check-prefix=LOWER

// The operands of the multiplication are mod reduced from the default scale,
// so both mod reduces get their input raised to (default scale + log q_1) = 90
// and produce the default scale again. The scale adjustments of the operations
// after them must be decided with those scales: one operand of the
// multiplication is raised by 10 bits to reach the target scale of level 0
// (45 + 55 = 100), and the plaintext added to the product must be encoded at
// the scale of the product.

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  // CHECK: func.func @mul_of_mod_reduced_operands
  // CHECK: %[[cst:.*]] = arith.constant dense<7.0
  // CHECK: mgmt.init %[[cst]] {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 100>}
  // LOWER: func.func @mul_of_mod_reduced_operands
  // LOWER: ckks.add_plain
  func.func @mul_of_mod_reduced_operands(%arg0: !secret.secret<tensor<8xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>}) -> !secret.secret<tensor<8xf32>> {
    %cst = arith.constant dense<7.0> : tensor<8xf32>
    %0 = secret.generic(%arg0 : !secret.secret<tensor<8xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>}) {
    ^body(%x: tensor<8xf32>):
      %w = arith.addf %x, %x {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xf32>
      // CHECK: mgmt.modreduce
      // CHECK-SAME: scale = 45
      %a = mgmt.modreduce %w {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<8xf32>
      %y = arith.subf %x, %x {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xf32>
      // CHECK: mgmt.modreduce
      // CHECK-SAME: scale = 45
      %b = mgmt.modreduce %y {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<8xf32>
      // CHECK: arith.mulf {{.*}}dimension = 3, scale = 100
      %m = arith.mulf %a, %b {mgmt.mgmt = #mgmt.mgmt<level = 0, dimension = 3>} : tensor<8xf32>
      // CHECK: mgmt.relinearize
      // CHECK-SAME: scale = 100
      %r = mgmt.relinearize %m {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<8xf32>
      %p = mgmt.init %cst {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<8xf32>
      // CHECK: arith.addf
      // CHECK-SAME: scale = 100
      %s = arith.addf %r, %p {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<8xf32>
      secret.yield %s : tensor<8xf32>
    } -> (!secret.secret<tensor<8xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    return %0 : !secret.secret<tensor<8xf32>>
  }
}
