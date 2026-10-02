// RUN: not heir-opt --generate-param-ckks="cheddar-parameter-file=%S/does_not_exist.json" %s 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not heir-opt --generate-param-ckks="cheddar-parameter-file=%s" %s 2>&1 | FileCheck %s --check-prefix=MALFORMED

// MISSING: cannot read CHEDDAR parameter file
// The file itself is not a parameter file; the error names it.
// MALFORMED: cheddar_parameter_file_unreadable.mlir:

module attributes {backend.cheddar, scheme.ckks} {
  func.func @add(%arg0: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) {
    return %arg0 : !secret.secret<f32>
  }
}
