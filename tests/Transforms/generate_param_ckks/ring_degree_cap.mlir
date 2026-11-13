// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" --generate-param-ckks --split-input-file --verify-diagnostics %s
// RUN: heir-opt --annotate-module="backend=openfhe scheme=ckks" --apply-config-override="config=max_ring_degree=65536" --generate-param-ckks --split-input-file --verify-diagnostics %s

// 36 levels of 45-bit primes need more than the 1747 bits of P * Q that the
// logN = 16 ring allows at 128-bit security.

// expected-error@below {{ring degree 131072 exceeds the backend's largest ring degree 65536}}
module {
  func.func @add(%arg0: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 36>}) -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 36>}) {
    %0 = secret.generic(%arg0: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 36>}) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 36>}} {
    ^body(%input0: f32):
      %1 = arith.addf %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 36>} : f32
      secret.yield %1 : f32
    } -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 36>})
    return %0 : !secret.secret<f32>
  }
}

// -----

// Parameters the module already carries are held to the same limit.

// expected-error@below {{ring degree 131072 exceeds the backend's largest ring degree 65536}}
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 17, Q = [36028797018652673], P = [1152921504606994433], logDefaultScale = 45>} {
  func.func @add(%arg0: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) {
    %0 = secret.generic(%arg0: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 0>}} {
    ^body(%input0: f32):
      %1 = arith.addf %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : f32
      secret.yield %1 : f32
    } -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    return %0 : !secret.secret<f32>
  }
}
