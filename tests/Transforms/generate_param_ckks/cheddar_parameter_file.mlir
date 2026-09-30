// RUN: heir-opt --generate-param-ckks="cheddar-parameter-file=%S/../../Parameters/cheddar/bootparam_35.json min-slot-count=1024" %s | FileCheck %s

// A two-level program on CHEDDAR's 32-bit bootstrapping chain. The file
// describes a single logN 16 ring; the scheme parameters model its modulus
// growth per level (about 50 bits for the two terminal primes of level 0, 35
// bits above), and the runtime parameter set records the file's own primes
// and its whole layout: CHEDDAR's Parameter requires the top level to hold
// every prime, so the program computes on the lowest three levels and
// encrypts at level 2. CHEDDAR's files carry no key-switching policy.

// CHECK: module attributes {
// CHECK-SAME: backend.cheddar, cheddar.parameter_set = #cheddar.parameter_set<logN = 16, logScale = 35
// CHECK-SAME: mainPrimes = [1060765697, 1209139201, 967180289, 1083703297, 1062469633,
// CHECK-SAME: auxPrimes = [2147352577, 2146959361, 2146041857,
// CHECK-SAME: terminalPrimes = [30539777, 37224449, 33292289, 32899073, 35389441]
// CHECK-SAME: levelConfig = [0, 2, 2, 1, 4, 0, 1, 5, 3, 4, 5, 3, 7, 2, 9, 1, 11, 0, 8, 5, 10, 4, 12, 3, 14, 2, 16, 1, 18, 0, 15, 5, 17, 4, 19, 3, 21, 2, 23, 1, 25, 1, 27, 1, 29, 1, 31, 1, 33, 1, 35, 1, 37, 1, 39, 1, 41, 1, 43, 1, 43, 3, 43, 5]
// CHECK-SAME: wordBits = 32
// CHECK-SAME: defaultEncryptionLevel = 2
// CHECK-NOT: defaultNumAux
// CHECK-SAME: denseHammingWeight = 32768
// CHECK-SAME: sparseHammingWeight = 32768>
// CHECK-SAME: ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [{{([0-9]{15,16})}}, {{([0-9]{11}(, [0-9]{11}){18})}}, {{([0-9]{18}(, [0-9]{18}){7})}}, {{([0-9]{19}(, [0-9]{19}){1})}}, {{([0-9]{15,16}(, [0-9]{15,16}){1})}}], P = [{{([0-9]{10}(, [0-9]{10}){11})}}], logDefaultScale = 35
// CHECK-SAME: mgmt.top_level = 2 : i64
// CHECK-SAME: scheme.actual_slot_count = 32768 : i64
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
