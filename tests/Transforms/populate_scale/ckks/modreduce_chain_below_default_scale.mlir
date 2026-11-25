// RUN: heir-opt %s --populate-scale-ckks | FileCheck %s

// A mod reduce whose input is at the default scale gets its input raised by
// an adjust_scale, so that its result is at the default scale again. The scale
// analysis that decides where to insert adjust_scale ops must model this:
// otherwise the values below such a mod reduce are assigned scales that are
// too small, and the operands of the addition below, which are one and two
// such mod reduces away from the input, appear to mismatch. That used to
// insert an adjust_scale with a target scale below the scale of its input
// ("delta scale is negative").

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797017456641, 35184365273089, 35184377331713, 35184366911489], P = [1152921504608747521], logDefaultScale = 45>, scheme.ckks} {
  // CHECK: func.func @modreduce_at_default_scale
  // CHECK-NOT: mgmt.adjust_scale
  // CHECK: arith.mulf
  // CHECK-SAME: scale = 90
  // CHECK-NEXT: mgmt.modreduce
  // CHECK-SAME: scale = 45
  // CHECK: arith.mulf
  // CHECK-SAME: scale = 90
  // CHECK-NEXT: mgmt.modreduce
  // CHECK-SAME: scale = 45
  // CHECK-NEXT: mgmt.level_reduce
  // CHECK-SAME: scale = 45
  // CHECK-NEXT: arith.addf
  // CHECK-SAME: scale = 45
  // CHECK-NOT: mgmt.adjust_scale
  // CHECK: return
  func.func @modreduce_at_default_scale(%arg0: !secret.secret<f32>) -> !secret.secret<f32> {
    %0 = secret.generic(%arg0 : !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 45>}) {
    ^body(%input0: f32):
      %a = arith.addf %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 3>} : f32
      %1 = mgmt.modreduce %a {mgmt.mgmt = #mgmt.mgmt<level = 2>} : f32
      %2 = arith.addf %1, %1 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : f32
      %3 = mgmt.modreduce %2 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : f32
      %4 = mgmt.level_reduce %1 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : f32
      %5 = arith.addf %4, %3 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : f32
      secret.yield %5 : f32
    } -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1>})
    return %0 : !secret.secret<f32>
  }
}
