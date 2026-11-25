// A 3x3 max pool with stride 2 and padding 1, in the form the torch importer
// produces: the padding and the initial value are negative infinity.
func.func @max_pool(%x: tensor<1x2x8x8xf32>) -> tensor<1x2x4x4xf32> {
  %neg_inf = arith.constant 0xFF800000 : f32
  %padded = tensor.pad %x low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %neg_inf : f32
  } : tensor<1x2x8x8xf32> to tensor<1x2x10x10xf32>
  %empty = tensor.empty() : tensor<1x2x4x4xf32>
  %init = linalg.fill ins(%neg_inf : f32) outs(%empty : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %window = tensor.empty() : tensor<3x3xf32>
  %out = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded, %window : tensor<1x2x10x10xf32>, tensor<3x3xf32>) outs(%init : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  return %out : tensor<1x2x4x4xf32>
}
