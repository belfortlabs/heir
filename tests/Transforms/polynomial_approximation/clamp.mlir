// RUN: heir-opt --split-input-file --polynomial-approximation %s | FileCheck %s
// RUN: heir-opt --split-input-file --polynomial-approximation=use-composite-relu=true %s | FileCheck %s --check-prefix=COMPOSITE

// ReLU6 = min(max(x, 0), 6) = max(x, 0) - max(x - 6, 0). Both ReLUs read x, the
// first on the domain of x and the second on that domain shifted by -6.
// CHECK: @relu6
// CHECK-SAME: (%[[X:.*]]: tensor<4xf32>
// CHECK-DAG: %[[C6:.*]] = arith.constant dense<6.000000e+00>
// CHECK-DAG: %[[R0:.*]] = polynomial.eval {{.*}}, %[[X]] {domain_lower = -4.000000e+00 : f64, domain_upper = 8.000000e+00 : f64}
// CHECK-DAG: %[[SHIFTED:.*]] = arith.subf %[[X]], %[[C6]]
// CHECK-DAG: %[[R6:.*]] = polynomial.eval {{.*}}, %[[SHIFTED]] {domain_lower = -1.000000e+01 : f64, domain_upper = 2.000000e+00 : f64}
// CHECK: %[[RES:.*]] = arith.subf %[[R0]], %[[R6]]
// CHECK-NOT: arith.maximumf
// CHECK-NOT: arith.minimumf
// CHECK: return %[[RES]]

// With composite ReLU, each ReLU is x * step(x / B) with B = max|domain|:
// 1/8 for max(x, 0) and 1/10 for max(x - 6, 0).
// COMPOSITE: @relu6
// COMPOSITE-DAG: arith.constant dense<1.250000e-01>
// COMPOSITE-DAG: arith.constant dense<1.000000e-01>
// COMPOSITE-COUNT-6: polynomial.eval
// COMPOSITE-NOT: arith.maximumf
// COMPOSITE-NOT: arith.minimumf
func.func @relu6(%x: tensor<4xf32> {secret.secret}) -> tensor<4xf32> {
  %c0 = arith.constant dense<0.0> : tensor<4xf32>
  %c6 = arith.constant dense<6.0> : tensor<4xf32>
  %0 = arith.maximumf %x, %c0 {domain_lower = -4.0 : f64, domain_upper = 8.0 : f64} : tensor<4xf32>
  %1 = arith.minimumf %c6, %0 {domain_lower = 0.0 : f64, domain_upper = 8.0 : f64} : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// max(min(x, 1), 0) is the same clamp with the halves in the other order (as
// in hard-sigmoid).
// CHECK: @clamp_min_first
// CHECK-SAME: (%[[X:.*]]: f32
// CHECK-DAG: %[[C1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG: %[[R0:.*]] = polynomial.eval {{.*}}, %[[X]] {domain_lower = -2.000000e+00 : f64, domain_upper = 3.000000e+00 : f64}
// CHECK-DAG: %[[SHIFTED:.*]] = arith.subf %[[X]], %[[C1]]
// CHECK-DAG: %[[R1:.*]] = polynomial.eval {{.*}}, %[[SHIFTED]] {domain_lower = -3.000000e+00 : f64, domain_upper = 2.000000e+00 : f64}
// CHECK: %[[RES:.*]] = arith.subf %[[R0]], %[[R1]]
// CHECK: return %[[RES]]
func.func @clamp_min_first(%x: f32 {secret.secret}) -> f32 {
  %c0 = arith.constant 0.0 : f32
  %c1 = arith.constant 1.0 : f32
  %0 = arith.minimumf %x, %c1 {domain_lower = -2.0 : f64, domain_upper = 3.0 : f64} : f32
  %1 = arith.maximumf %0, %c0 : f32
  return %1 : f32
}

// -----

// Without a domain, x is assumed to be in [-1, 1], where max(x - 6, 0) is zero,
// so only max(x, 0) is approximated.
// CHECK: @relu6_default_domain
// CHECK-SAME: (%[[X:.*]]: tensor<4xf32>
// CHECK: %[[R0:.*]] = polynomial.eval {{.*}}, %[[X]] {domain_lower = -1.000000e+00 : f64, domain_upper = 1.000000e+00 : f64}
// CHECK-NOT: polynomial.eval
// CHECK-NOT: arith.subf
// CHECK: return %[[R0]]
func.func @relu6_default_domain(%x: tensor<4xf32> {secret.secret}) -> tensor<4xf32> {
  %c0 = arith.constant dense<0.0> : tensor<4xf32>
  %c6 = arith.constant dense<6.0> : tensor<4xf32>
  %0 = arith.maximumf %x, %c0 : tensor<4xf32>
  %1 = arith.minimumf %0, %c6 : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// A nonzero lower bound adds it back: clamp(x, -1, 1) = -1 + max(x + 1, 0) - max(x - 1, 0).
// CHECK: @hardtanh
// CHECK-SAME: (%[[X:.*]]: f32
// CHECK-DAG: %[[CM1:.*]] = arith.constant -1.000000e+00 : f32
// CHECK-DAG: %[[C1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG: %[[SLO:.*]] = arith.subf %[[X]], %[[CM1]]
// CHECK-DAG: %[[RLO:.*]] = polynomial.eval {{.*}}, %[[SLO]] {domain_lower = -1.000000e+00 : f64, domain_upper = 3.000000e+00 : f64}
// CHECK-DAG: %[[SHI:.*]] = arith.subf %[[X]], %[[C1]]
// CHECK-DAG: %[[RHI:.*]] = polynomial.eval {{.*}}, %[[SHI]] {domain_lower = -3.000000e+00 : f64, domain_upper = 1.000000e+00 : f64}
// CHECK: %[[DIFF:.*]] = arith.subf %[[RLO]], %[[RHI]]
// CHECK: %[[RES:.*]] = arith.addf %[[DIFF]], %[[CM1]]
// CHECK: return %[[RES]]
func.func @hardtanh(%x: f32 {secret.secret}) -> f32 {
  %cm1 = arith.constant -1.0 : f32
  %c1 = arith.constant 1.0 : f32
  %0 = arith.maximumf %x, %cm1 {domain_lower = -2.0 : f64, domain_upper = 2.0 : f64} : f32
  %1 = arith.minimumf %0, %c1 : f32
  return %1 : f32
}

// -----

// Not a clamp: the interval is empty.
// CHECK: @empty_interval
// CHECK-NOT: arith.subf
func.func @empty_interval(%x: f32 {secret.secret}) -> f32 {
  %c0 = arith.constant 0.0 : f32
  %c1 = arith.constant 1.0 : f32
  %0 = arith.maximumf %x, %c1 : f32
  %1 = arith.minimumf %0, %c0 : f32
  return %1 : f32
}

// -----

// A plaintext clamp stays exact: without a domain, the split would assume x in
// [-1, 1] and drop the upper bound.
// CHECK: @plaintext_clamp
// CHECK-SAME: (%[[X:.*]]: f32)
// CHECK: %[[MAX:.*]] = arith.maximumf %[[X]]
// CHECK: %[[MIN:.*]] = arith.minimumf %[[MAX]]
// CHECK-NOT: polynomial.eval
// CHECK: return %[[MIN]]
// COMPOSITE: @plaintext_clamp
// COMPOSITE: arith.maximumf
// COMPOSITE: arith.minimumf
func.func @plaintext_clamp(%x: f32) -> f32 {
  %c0 = arith.constant 0.0 : f32
  %c6 = arith.constant 6.0 : f32
  %0 = arith.maximumf %x, %c0 : f32
  %1 = arith.minimumf %0, %c6 : f32
  return %1 : f32
}

// -----

// The inner max has another user, so it is approximated anyway: the clamp is
// not split, because the split would add two ReLUs instead of one minimum.
// CHECK: @inner_has_other_user
// CHECK-NOT: arith.subf
// CHECK: return
func.func @inner_has_other_user(%x: f32 {secret.secret}) -> (f32, f32) {
  %c0 = arith.constant 0.0 : f32
  %c6 = arith.constant 6.0 : f32
  %0 = arith.maximumf %x, %c0 {domain_lower = -4.0 : f64, domain_upper = 8.0 : f64} : f32
  %1 = arith.minimumf %0, %c6 {domain_lower = 0.0 : f64, domain_upper = 8.0 : f64} : f32
  return %0, %1 : f32, f32
}
