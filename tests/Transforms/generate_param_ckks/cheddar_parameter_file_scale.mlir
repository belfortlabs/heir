// RUN: heir-opt --generate-param-ckks="cheddar-parameter-file=%S/../../Parameters/cheddar/bootparam_35.json scaling-mod-bits=40" --verify-diagnostics %s

// The file fixes the scale; an explicitly requested scale must match it.
// expected-error@below {{scaling-mod-bits=40 conflicts with the 35-bit scale of}}
module attributes {backend.cheddar, scheme.ckks} {
  func.func @add(%arg0: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) {
    return %arg0 : !secret.secret<f32>
  }
}
