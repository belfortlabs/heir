// RUN: heir-opt --generate-param-ckks="cheddar-parameter-file=%S/../../Parameters/cyclops/bootparam_35bit_14lvl.json min-slot-count=8" %s | FileCheck %s

// A bootstrapping program takes the largest ring, whose profile carries the
// bootstrap chain: 15 residual levels plus 3 StC + 8 EvalMod + 5 CtS levels
// above them, and the file's bootstrap split. The model chain has one modulus
// per level; the EvalMod levels grow the modulus by about 60 bits. The
// bootstrap lands at level 14, so a program that is only 12 levels deep is
// placed at the top of the chain, and the top is pinned for later level
// annotation.

// CHECK: module attributes {
// CHECK-SAME: backend.cheddar
// CHECK-SAME: cheddar.bootstrap_config = #cheddar.bootstrap_config<numCtsLevels = 5, numStcLevels = 3, numEvalModLevels = 8, logMessageRatio = 12, evalMod = <type = "cos_hk_even", degree = 26, interval = 16, logIntervalReduction = 3, invDegree = 0>>
// CHECK-SAME: cheddar.parameter_set = #cheddar.parameter_set<logN = 16, logScale = 35
// CHECK-SAME: levelConfig = [0, 2, 2, 1, 4, 0, 1, 5, 3, 4, 5, 3, 7, 2, 9, 1, 11, 0, 8, 5, 10, 4, 12, 3, 14, 2, 16, 1, 18, 0, 15, 5, 17, 4, 19, 3, 21, 3, 23, 3, 25, 3, 27, 3, 29, 3, 31, 3, 33, 3, 35, 3, 37, 3, 39, 3, 41, 3, 43, 3, 43, 5]
// CHECK-SAME: wordBits = 32
// CHECK-SAME: defaultEncryptionLevel = 14
// CHECK-SAME: maxLogPq = 1.730000e+03 : f64
// CHECK-SAME: denseHammingWeight = 1024
// CHECK-SAME: sparseHammingWeight = 32>
// CHECK-SAME: ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [{{([0-9]{15,16})}}, {{([0-9]{11}(, [0-9]{11}){16})}}, {{([0-9]{18,19}(, [0-9]{18,19}){11})}}, {{([0-9]{15,16})}}]
// CHECK-SAME: mgmt.top_level = 14 : i64
// CHECK-SAME: scheme.actual_slot_count = 32768 : i64
// CHECK-SAME: scheme.requested_slot_count = 8 : i64
module attributes {backend.cheddar, scheme.ckks} {
  // CHECK: func.func @bootstrap(%{{.*}}: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 14>}) -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 12>})
  // CHECK: mgmt.bootstrap
  // CHECK-SAME: mgmt.mgmt = #mgmt.mgmt<level = 14>
  // CHECK: mgmt.modreduce
  // CHECK-SAME: mgmt.mgmt = #mgmt.mgmt<level = 13>
  // CHECK: mgmt.modreduce
  // CHECK-SAME: mgmt.mgmt = #mgmt.mgmt<level = 12>
  func.func @bootstrap(%arg0: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 12>}) -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 10>}) {
    %0 = secret.generic(%arg0: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 12>}) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 12>}} {
    ^body(%input: f32):
      %1 = mgmt.bootstrap %input {mgmt.mgmt = #mgmt.mgmt<level = 12>} : f32
      %2 = mgmt.modreduce %1 {mgmt.mgmt = #mgmt.mgmt<level = 11>} : f32
      %3 = mgmt.modreduce %2 {mgmt.mgmt = #mgmt.mgmt<level = 10>} : f32
      secret.yield %3 : f32
    } -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 10>})
    return %0 : !secret.secret<f32>
  }
}
