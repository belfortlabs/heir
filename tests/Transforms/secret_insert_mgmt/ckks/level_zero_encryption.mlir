// RUN: heir-opt --mlir-to-secret-arithmetic --secret-insert-mgmt-ckks="level-budget=2 level-zero-encryption=true" %s | FileCheck %s
// RUN: heir-opt --mlir-to-ckks="greedy-level-budget=2 level-zero-encryption=true" %s | FileCheck %s --check-prefix=PIPELINE

// The client encrypts at the bottom of the modulus chain, so the entry
// argument is annotated level 0 and the first thing the computation does is
// bootstrap it back to the top. The encryption helper inherits the argument's
// level, which is what makes the client encode a single RNS limb.

// CHECK: func.func @level_zero_encryption
// CHECK-SAME: %[[arg0:.*]]: !secret.secret<tensor<1x1024xf16>> {mgmt.level_zero_arg, mgmt.mgmt = #mgmt.mgmt<level = 0>
// CHECK:   secret.generic(%[[arg0]]: !secret.secret<tensor<1x1024xf16>> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
// CHECK:   (%[[input0:.*]]: tensor<1x1024xf16>):
// CHECK:     %[[boot:.*]] = mgmt.bootstrap %[[input0]] {mgmt.mgmt = #mgmt.mgmt<level = 2>}
// CHECK:     %[[v1:.*]] = arith.addf %[[boot]], %[[boot]] {mgmt.mgmt = #mgmt.mgmt<level = 2>}
// CHECK:     %[[v2:.*]] = mgmt.modreduce %[[v1]] {mgmt.mgmt = #mgmt.mgmt<level = 1>}
// CHECK:     %[[v3:.*]] = arith.addf %[[v2]], %[[v2]] {mgmt.mgmt = #mgmt.mgmt<level = 1>}
// CHECK:     %[[v4:.*]] = mgmt.modreduce %[[v3]] {mgmt.mgmt = #mgmt.mgmt<level = 0>}
// CHECK:     %[[v5:.*]] = arith.addf %[[v4]], %[[v4]] {mgmt.mgmt = #mgmt.mgmt<level = 0>}
// CHECK:     secret.yield %[[v5]]

// CHECK: func.func @level_zero_encryption__encrypt__arg0
// CHECK:   secret.conceal %{{.*}} {mgmt.mgmt = #mgmt.mgmt<level = 0>}

// Parameter generation reads the maximum level off the whole program: the
// deepest value is no longer the entry argument, so a chain sized from the
// arguments alone would collapse to a single prime.
// PIPELINE: #modulus_chain_L2_C0 = #lwe.modulus_chain<elements = <[[q0:[0-9]+]] : i64, [[q1:[0-9]+]] : i64, [[q2:[0-9]+]] : i64>, current = 0>
// PIPELINE: ckks.schemeParam = #ckks.scheme_param<logN = {{[0-9]+}}, Q = [[[q0]], [[q1]], [[q2]]]
// PIPELINE: ckks.bootstrap %{{.*}} : !ct_L0 -> !ct_L2
// PIPELINE: func.func @level_zero_encryption(%{{.*}}: tensor<1x!ct_L0>
// PIPELINE: func.func @level_zero_encryption__encrypt__arg0
// PIPELINE:   lwe.rlwe_encode %{{.*}} {{{.*}}level = 0 : i64

module attributes {backend.lattigo, scheme.ckks, backend.config_override = {bootstrapLevelsConsumed = 0 : i32}} {
  func.func @level_zero_encryption(
      %x : f16 {secret.secret}
    ) -> f16 {
      %0 = arith.addf %x, %x : f16
      %r0 = mgmt.modreduce %0 : f16
      %1 = arith.addf %r0, %r0 : f16
      %r1 = mgmt.modreduce %1 : f16
      %2 = arith.addf %r1, %r1 : f16
      return %2 : f16
  }
}
