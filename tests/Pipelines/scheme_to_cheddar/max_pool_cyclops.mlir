// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" --torch-linalg-to-ckks="min-slot-count=256" %s | FileCheck %s
// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" --torch-linalg-to-ckks="min-slot-count=256" --scheme-to-cheddar="entry-function=maxpool1d runtime=cyclops" %s | FileCheck %s --check-prefix=KEYS
// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" --torch-linalg-to-ckks="min-slot-count=32" --scheme-to-cheddar="entry-function=maxpool1d runtime=cyclops" %s | FileCheck %s --check-prefix=SMALL

// A MaxPool1d(2) over 4 channels of 7 values with the domain [-2, 6]:
// - linalg-canonicalizations slices off the last value of each channel,
//   which no window reads; it does not normalize the input, because Cyclops
//   scales by value_bound = max(|-2|, |6|) = 6;
// - the 4 channels of 6 values lie end to end, and one kernel.max_pool
//   pools all of them;
// - the pool drops 4 levels: the scaling into the comparison domain (no
//   gather stage carries it), the league's multiplication by the selector and
//   2 compaction stages. The league needs a bootstrap end level of at least
//   11, so the chain tops out at 11.

// CHECK: func.func @maxpool1d__preprocessed
// CHECK: ckks.bootstrap %{{.*}} : !ct_L3 -> !ct_L11
// CHECK: kernel.max_pool %{{.*}} {input_length = 24 : i64, num_slots = 256 : i64, stride = 2 : i64, value_bound = 6.000000e+00 : f64, window_size = 2 : i64} : tensor<1x!ct_L11> -> tensor<1x!ct_L7>

// Key planning pins the output level of each pool to HEIR's level drop, and
// checks that Cyclops' MaxPool runs from the planned input level.
// KEYS: func.func @maxpool1d__setup
// KEYS-SAME: cheddar.evaluation_keys

// With 32 slots, the league bootstraps with the 256-slot transform that the
// boot context prepares, and the key planner requests its keys.
// SMALL: cheddar.max_pool
// SMALL-SAME: levelConsumption = 4 : i64, num_slots = 32 : i64
// SMALL: func.func @maxpool1d__setup
// SMALL-SAME: cheddar.evaluation_keys

module {
  func.func @maxpool1d(%arg0: tensor<1x4x7xf32> {secret.secret}) -> tensor<1x4x3xf32> {
    %neg_inf = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<1x4x3xf32>
    %1 = linalg.fill ins(%neg_inf : f32) outs(%0 : tensor<1x4x3xf32>) -> tensor<1x4x3xf32>
    %window = tensor.empty() : tensor<2xf32>
    %2 = linalg.pooling_ncw_max {dilations = dense<1> : vector<1xi64>, domain_lower = -2.0 : f64, domain_upper = 6.0 : f64, strides = dense<2> : vector<1xi64>} ins(%arg0, %window : tensor<1x4x7xf32>, tensor<2xf32>) outs(%1 : tensor<1x4x3xf32>) -> tensor<1x4x3xf32>
    return %2 : tensor<1x4x3xf32>
  }
}
