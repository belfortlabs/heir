// RUN: heir-opt --annotate-module="backend=lattigo scheme=bfv" --apply-config-override="config=max_ring_degree=1024" --generate-param-bfv --split-input-file --verify-diagnostics %s

// Even the smallest secure ring for this chain exceeds a 1024 limit.

// expected-error@below {{exceeds the backend's largest ring degree 1024}}
module {
  func.func @add(%arg0: !secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) -> (!secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) {
    %0 = secret.generic(%arg0: !secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 0>}} {
    ^body(%input0: i16):
      %1 = arith.addi %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : i16
      secret.yield %1 : i16
    } -> (!secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    return %0 : !secret.secret<i16>
  }
}

// -----

// Parameters the module already carries are held to the same limit.

// expected-error@below {{ring degree 4096 exceeds the backend's largest ring degree 1024}}
module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 12, Q = [2147565569], P = [2147573761], plaintextModulus = 65537>} {
  func.func @add(%arg0: !secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) -> (!secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) {
    %0 = secret.generic(%arg0: !secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 0>}} {
    ^body(%input0: i16):
      %1 = arith.addi %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : i16
      secret.yield %1 : i16
    } -> (!secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    return %0 : !secret.secret<i16>
  }
}
