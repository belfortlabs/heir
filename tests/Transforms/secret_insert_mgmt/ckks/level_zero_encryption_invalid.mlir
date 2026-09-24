// Level-zero encryption bootstraps every entry ciphertext, so it only applies
// where bootstraps are placed: the greedy CKKS path. Asking for it anywhere
// else is rejected rather than ignored.

// RUN: not heir-opt --mlir-to-bgv="level-zero-encryption=true" %s 2>&1 | FileCheck %s
// RUN: not heir-opt --mlir-to-ckks="level-zero-encryption=true ciphertext-management-style=orbit-ilp" %s 2>&1 | FileCheck %s

// CHECK: level-zero-encryption requires the CKKS scheme with greedy ciphertext management

func.func @rejects_non_greedy_ckks(%x : i16 {secret.secret}) -> i16 {
  %0 = arith.addi %x, %x : i16
  return %0 : i16
}
