// RUN: heir-opt --generate-param-ckks="cheddar-parameter-file=%S/../../Parameters/cyclops/bootparam_35bit_14lvl.json min-slot-count=1024" %s | FileCheck %s

// A two-level program on a 32-bit Cyclops chain. The smallest ring that holds
// three levels at 1024 slots is logN 13 (its profile holds three residual
// levels); the scheme parameters model the file's modulus growth per level
// (about 50 bits for the two terminal primes of level 0, 35 bits above), and
// the runtime parameter set records the file's own primes and layout, trimmed
// to the levels the program uses.

// CHECK: module attributes {
// CHECK-SAME: backend.cheddar, cheddar.parameter_set = #cheddar.parameter_set<logN = 13, logScale = 35
// CHECK-SAME: mainPrimes = [969146369, 1172439041, 958136321, 1070727169, 1115815937,
// CHECK-SAME: auxPrimes = [2147352577,
// CHECK-SAME: terminalPrimes = [30539777, 32899073, 29884417, 31326209, 36175873]
// CHECK-SAME: levelConfig = [0, 2, 2, 1, 4, 0]
// CHECK-SAME: wordBits = 32
// CHECK-SAME: defaultEncryptionLevel = 2
// CHECK-SAME: defaultNumAux = 12
// CHECK-SAME: levelSpecificKs = true
// CHECK-SAME: maxKeySwitchAux = 23
// CHECK-SAME: maxLogPq = 1.840000e+02 : f64
// CHECK-SAME: denseHammingWeight = 128
// CHECK-SAME: sparseHammingWeight = 128>
// CHECK-SAME: ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [{{([0-9]{15,16})}}, {{([0-9]{11})}}, {{([0-9]{11})}}], P = [{{([0-9]{9,10}(, [0-9]{9,10}){22})}}], logDefaultScale = 35
// CHECK-SAME: encryptionTechnique = extended
// CHECK-SAME: mgmt.top_level = 2 : i64
// CHECK-SAME: scheme.actual_slot_count = 4096 : i64
// CHECK-SAME: scheme.requested_slot_count = 1024 : i64
module attributes {backend.cheddar, scheme.ckks} {
  func.func @square_twice(%arg0: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 2>}) -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) {
    %0 = secret.generic(%arg0: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 2>}) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 2>}} {
    ^body(%input0: f32):
      %1 = arith.mulf %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : f32
      %2 = mgmt.relinearize %1 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : f32
      %3 = mgmt.modreduce %2 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : f32
      %4 = arith.mulf %3, %3 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : f32
      %5 = mgmt.relinearize %4 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : f32
      %6 = mgmt.modreduce %5 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : f32
      secret.yield %6 : f32
    } -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    return %0 : !secret.secret<f32>
  }
}
