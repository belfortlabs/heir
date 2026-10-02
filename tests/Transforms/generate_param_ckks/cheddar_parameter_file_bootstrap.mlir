// RUN: heir-opt --generate-param-ckks="cheddar-parameter-file=%S/../../Parameters/cheddar/bootparam_35.json min-slot-count=8" %s | FileCheck %s

// A bootstrapping program on CHEDDAR's chain takes all 32 levels. The file's
// default encryption level (19) is where SlotToCoeff starts, with 4 CtS and 8
// EvalMod levels above it, so the bootstrap lands 3 StC levels lower, at
// level 16: a program that is only 12 levels deep is placed at the top of
// that, and the top is pinned for later level annotation. The file states no
// message ratio, so it is derived from its 50.0-bit q0 over the 35-bit
// scale: floor(15.0 - 2) = 13.

// CHECK: module attributes {
// CHECK-SAME: backend.cheddar
// CHECK-SAME: cheddar.bootstrap_config = #cheddar.bootstrap_config<numCtsLevels = 4, numStcLevels = 3, numEvalModLevels = 8, logMessageRatio = 13>
// CHECK-SAME: cheddar.parameter_set = #cheddar.parameter_set<logN = 16, logScale = 35
// CHECK-SAME: levelConfig = [0, 2, 2, 1, 4, 0, 1, 5, 3, 4, 5, 3, 7, 2, 9, 1, 11, 0, 8, 5, 10, 4, 12, 3, 14, 2, 16, 1, 18, 0, 15, 5, 17, 4, 19, 3, 21, 2, 23, 1, 25, 1, 27, 1, 29, 1, 31, 1, 33, 1, 35, 1, 37, 1, 39, 1, 41, 1, 43, 1, 43, 3, 43, 5]
// CHECK-SAME: wordBits = 32
// CHECK-SAME: defaultEncryptionLevel = 19
// CHECK-SAME: denseHammingWeight = 32768
// CHECK-SAME: sparseHammingWeight = 32>
// CHECK-SAME: ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [{{([0-9]{15,16})}}, {{([0-9]{11}(, [0-9]{11}){18})}}, {{([0-9]{18}(, [0-9]{18}){7})}}, {{([0-9]{19}(, [0-9]{19}){1})}}, {{([0-9]{15,16}(, [0-9]{15,16}){1})}}]
// CHECK-SAME: mgmt.top_level = 16 : i64
// CHECK-SAME: scheme.actual_slot_count = 32768 : i64
// CHECK-SAME: scheme.requested_slot_count = 8 : i64
module attributes {backend.cheddar, scheme.ckks} {
  // CHECK: func.func @bootstrap(%{{.*}}: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 16>}) -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 14>})
  // CHECK: mgmt.bootstrap
  // CHECK-SAME: mgmt.mgmt = #mgmt.mgmt<level = 16>
  // CHECK: mgmt.modreduce
  // CHECK-SAME: mgmt.mgmt = #mgmt.mgmt<level = 15>
  // CHECK: mgmt.modreduce
  // CHECK-SAME: mgmt.mgmt = #mgmt.mgmt<level = 14>
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
