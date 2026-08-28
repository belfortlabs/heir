// RUN: heir-opt --annotate-module="backend=openfhe scheme=ckks" %s | FileCheck %s
// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" %s | FileCheck %s --check-prefix=CHECK-CHEDDAR
// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks cheddar-runtime=cyclops" %s | FileCheck %s --check-prefix=CHECK-CYCLOPS
// RUN: not heir-opt --annotate-module="cheddar-runtime=bogus" %s 2>&1 | FileCheck %s --check-prefix=CHECK-BAD-RUNTIME

// CHECK: module attributes {backend.openfhe, scheme.ckks}
// CHECK-CHEDDAR: module attributes {backend.cheddar, scheme.ckks}
// The runtime attribute is what cheddar-to-emitc reads to pick which of the two
// CHEDDAR C++ APIs it emits against.
// CHECK-CYCLOPS: module attributes {backend.cheddar, cheddar.runtime.cyclops, scheme.ckks}
// CHECK-BAD-RUNTIME: Unknown CHEDDAR runtime: bogus
module {

}
