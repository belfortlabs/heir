// RUN: heir-opt --annotate-module="backend=lattigo scheme=ckks" --apply-config-override="config=max_ring_degree=256" --layout-propagation="min-slot-count=128" --verify-diagnostics %s

// The second conv reads the pixel-shuffled result of the first, so its result
// is shuffled by a gap of 2 and its 13 output channels reserve 4 blocks of
// 2x2: the filter matrix has 16 * 3 * 3 = 144 rows, more than the 128 slots of
// the 256 ring. Sized as dense data, it has only 13 * 3 * 3 = 117 rows.
func.func @gapped_conv(%arg0: !secret.secret<tensor<1x1x10x10xf32>>) -> !secret.secret<tensor<1x13x3x3xf32>> {
  %cst1 = arith.constant dense<0.0> : tensor<1x1x5x5xf32>
  %filter1 = arith.constant dense<1.0> : tensor<1x1x2x2xf32>
  %cst2 = arith.constant dense<0.0> : tensor<1x13x3x3xf32>
  %filter2 = arith.constant dense<1.0> : tensor<13x1x3x3xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1x10x10xf32>>) {
  ^body(%input0: tensor<1x1x10x10xf32>):
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%input0, %filter1 : tensor<1x1x10x10xf32>, tensor<1x1x2x2xf32>)
      outs(%cst1 : tensor<1x1x5x5xf32>) -> tensor<1x1x5x5xf32>
    // expected-error@+1 {{needs 144 slots, but the backend's largest ring degree 256 holds 128}}
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%1, %filter2 : tensor<1x1x5x5xf32>, tensor<13x1x3x3xf32>)
      outs(%cst2 : tensor<1x13x3x3xf32>) -> tensor<1x13x3x3xf32>
    secret.yield %2 : tensor<1x13x3x3xf32>
  } -> !secret.secret<tensor<1x13x3x3xf32>>
  return %0 : !secret.secret<tensor<1x13x3x3xf32>>
}
