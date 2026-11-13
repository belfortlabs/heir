// RUN: heir-opt --generate-param-ckks="cheddar-parameter-file=%S/../../Parameters/cheddar/bootparam_35.json" --split-input-file --verify-diagnostics %s

// A CHEDDAR parameter file describes the Cheddar backend's runtime.
// expected-error@below {{cheddar-parameter-file requires a Cheddar backend module}}
module attributes {backend.lattigo, scheme.ckks} {
  func.func @add(%arg0: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) {
    return %arg0 : !secret.secret<f32>
  }
}

// -----

// The file's ring holds 19 levels without bootstrapping.
// expected-error@below {{no ring profile in}}
module attributes {backend.cheddar, scheme.ckks} {
  func.func @deep(%arg0: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 20>}) -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 20>}) {
    return %arg0 : !secret.secret<f32>
  }
}
