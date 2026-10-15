// A secret per-channel scale multiplied into a secret feature map that spans
// two ciphertexts at 1024 slots.
#id = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#channel = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
module {
  func.func @squeeze_excite_multi_ct(%x : tensor<1x8x16x16xf32> {secret.secret}, %s : tensor<1x8x1x1xf32> {secret.secret}) -> tensor<1x8x16x16xf32> {
    %0 = tensor.empty() : tensor<1x8x16x16xf32>
    %1 = linalg.generic {indexing_maps = [#channel, #id, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%s, %x : tensor<1x8x1x1xf32>, tensor<1x8x16x16xf32>) outs(%0 : tensor<1x8x16x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.mulf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<1x8x16x16xf32>
    return %1 : tensor<1x8x16x16xf32>
  }
}
