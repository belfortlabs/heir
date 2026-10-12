// A secret per-channel scale multiplied into the output of a stride-2
// convolution, whose layout leaves gaps between the pixels of each channel.
#id = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#channel = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
module {
  func.func @squeeze_excite_strided(%x : tensor<1x1x8x8xf32> {secret.secret}, %s : tensor<1x4x1x1xf32> {secret.secret}) -> tensor<1x4x4x4xf32> {
    %filter = arith.constant dense<[[[[1.0]]], [[[2.0]]], [[[-1.0]]], [[[0.5]]]]> : tensor<4x1x1x1xf32>
    %init = arith.constant dense<0.0> : tensor<1x4x4x4xf32>
    %conv = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%x, %filter : tensor<1x1x8x8xf32>, tensor<4x1x1x1xf32>) outs(%init : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    %0 = tensor.empty() : tensor<1x4x4x4xf32>
    %1 = linalg.generic {indexing_maps = [#channel, #id, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%s, %conv : tensor<1x4x1x1xf32>, tensor<1x4x4x4xf32>) outs(%0 : tensor<1x4x4x4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.mulf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<1x4x4x4xf32>
    return %1 : tensor<1x4x4x4xf32>
  }
}
