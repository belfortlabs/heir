// RUN: heir-opt --generate-param-ckks="cheddar-parameter-file=%S/../../Parameters/cyclops/bootparam_35bit_19lvl.json min-slot-count=1024" %s | FileCheck %s

// A two-level program on a 32-bit Cyclops chain whose level 0 holds four
// terminal primes, about 100 bits. One model prime cannot have that size, so
// the model's level 0 is capped at 60 bits, the largest prime the model
// generates; the levels above it model the file's 35-bit growth. The runtime
// parameter set keeps the file's own primes and layout.

// CHECK: module attributes {
// CHECK-SAME: backend.cheddar, cheddar.parameter_set = #cheddar.parameter_set<logN = 14, logScale = 35
// CHECK-SAME: terminalPrimes = [36175873, 29884417, 32899073, 31326209, 37224449]
// CHECK-SAME: levelConfig = [0, 4, 2, 3, 4, 2]
// CHECK-SAME: wordBits = 32
// CHECK-SAME: defaultEncryptionLevel = 2
// CHECK-SAME: ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [{{([0-9]{19})}}, {{([0-9]{11})}}, {{([0-9]{11})}}], P = [{{([0-9]{9,10}(, [0-9]{9,10}){22})}}], logDefaultScale = 35
// CHECK-SAME: mgmt.top_level = 2 : i64
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
