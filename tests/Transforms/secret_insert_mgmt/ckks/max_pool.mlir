// RUN: heir-opt --secret-insert-mgmt-ckks="level-budget=7" %s | FileCheck %s --check-prefix=BOOT
// RUN: heir-opt --secret-insert-mgmt-ckks="level-budget=8" %s | FileCheck %s --check-prefix=NOBOOT

// A window of 16 slots drops 4 levels (two league multiplications and two
// compaction stages), but its league needs 7 levels in its input. After the
// multiplication and its rescale, one level is consumed: 1 + 7 levels exceed a
// budget of 7, so a bootstrap goes before the pool, and fit a budget of 8.
//
// The league also needs a bootstrap end level of at least 11, so the levels
// are shifted until the top level is 11.

// BOOT: func.func @wide_window
// BOOT-SAME: level = 11
// BOOT: mgmt.modreduce
// BOOT-SAME: level = 10
// BOOT: %[[boot:.*]] = mgmt.bootstrap
// BOOT-SAME: level = 11
// BOOT: kernel.max_pool %[[boot]]
// BOOT-SAME: level = 7

// NOBOOT: func.func @wide_window
// NOBOOT-SAME: level = 11
// NOBOOT-NOT: mgmt.bootstrap
// NOBOOT: %[[rescaled:.*]] = mgmt.modreduce
// NOBOOT-SAME: level = 10
// NOBOOT-NOT: mgmt.bootstrap
// NOBOOT: kernel.max_pool %[[rescaled]]
// NOBOOT-SAME: level = 6

module attributes {backend.cheddar, scheme.ckks} {
  func.func @wide_window(%arg0: !secret.secret<tensor<1x64xf32>>) -> !secret.secret<tensor<1x64xf32>> {
    %0 = secret.generic(%arg0 : !secret.secret<tensor<1x64xf32>>) {
    ^body(%x: tensor<1x64xf32>):
      %1 = arith.mulf %x, %x : tensor<1x64xf32>
      %2 = kernel.max_pool %1 {num_slots = 64 : i64, input_length = 64 : i64, window_size = 16 : i64, stride = 16 : i64, value_bound = 0.5 : f64} : tensor<1x64xf32> -> tensor<1x64xf32>
      secret.yield %2 : tensor<1x64xf32>
    } -> !secret.secret<tensor<1x64xf32>>
    return %0 : !secret.secret<tensor<1x64xf32>>
  }
}
