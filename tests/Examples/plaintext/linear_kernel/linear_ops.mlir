// Operations that are linear in a secret tensor, with cleartext weights
// passed as arguments so the test can choose them.

func.func @conv_padded(%x: tensor<1x4x8x8xf32> {secret.secret}, %w: tensor<4x4x3x3xf32>, %bias: tensor<4xf32>) -> tensor<1x4x8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %padded = tensor.pad %x low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %zero : f32
  } : tensor<1x4x8x8xf32> to tensor<1x4x10x10xf32>
  %empty = tensor.empty() : tensor<1x4x8x8xf32>
  %init = linalg.broadcast ins(%bias : tensor<4xf32>) outs(%empty : tensor<1x4x8x8xf32>) dimensions = [0, 2, 3]
  %out = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded, %w : tensor<1x4x10x10xf32>, tensor<4x4x3x3xf32>) outs(%init : tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32>
  return %out : tensor<1x4x8x8xf32>
}

func.func @conv_strided(%x: tensor<1x2x8x8xf32> {secret.secret}, %w: tensor<4x2x3x3xf32>) -> tensor<1x4x4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %padded = tensor.pad %x low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %zero : f32
  } : tensor<1x2x8x8xf32> to tensor<1x2x10x10xf32>
  %init = arith.constant dense<0.0> : tensor<1x4x4x4xf32>
  %out = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded, %w : tensor<1x2x10x10xf32>, tensor<4x2x3x3xf32>) outs(%init : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  return %out : tensor<1x4x4x4xf32>
}

func.func @matvec(%x: tensor<16xf32> {secret.secret}, %m: tensor<10x16xf32>) -> tensor<10xf32> {
  %init = arith.constant dense<0.0> : tensor<10xf32>
  %out = linalg.matvec ins(%m, %x : tensor<10x16xf32>, tensor<16xf32>) outs(%init : tensor<10xf32>) -> tensor<10xf32>
  return %out : tensor<10xf32>
}

func.func @sum_pool(%x: tensor<1x2x8x8xf32> {secret.secret}) -> tensor<1x2x4x4xf32> {
  %init = arith.constant dense<0.0> : tensor<1x2x4x4xf32>
  %window = tensor.empty() : tensor<2x2xf32>
  %out = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%x, %window : tensor<1x2x8x8xf32>, tensor<2x2xf32>) outs(%init : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  return %out : tensor<1x2x4x4xf32>
}

func.func @conv_dilated(%x: tensor<1x2x8x8xf32> {secret.secret}, %w: tensor<3x2x3x3xf32>) -> tensor<1x3x4x4xf32> {
  %init = arith.constant dense<0.0> : tensor<1x3x4x4xf32>
  %out = linalg.conv_2d_nchw_fchw {dilations = dense<2> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%x, %w : tensor<1x2x8x8xf32>, tensor<3x2x3x3xf32>) outs(%init : tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
  return %out : tensor<1x3x4x4xf32>
}

func.func @conv_depthwise(%x: tensor<1x4x8x8xf32> {secret.secret}, %w: tensor<4x3x3xf32>) -> tensor<1x4x3x3xf32> {
  %init = arith.constant dense<0.0> : tensor<1x4x3x3xf32>
  %out = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%x, %w : tensor<1x4x8x8xf32>, tensor<4x3x3xf32>) outs(%init : tensor<1x4x3x3xf32>) -> tensor<1x4x3x3xf32>
  return %out : tensor<1x4x3x3xf32>
}

// A grouped convolution in the form the torch importer produces: the padded
// input is split into groups by a reshape, and the result is merged again.
func.func @conv_grouped(%x: tensor<1x4x6x6xf32> {secret.secret}, %w: tensor<2x3x2x3x3xf32>) -> tensor<1x6x6x6xf32> {
  %zero = arith.constant 0.0 : f32
  %padded = tensor.pad %x low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %zero : f32
  } : tensor<1x4x6x6xf32> to tensor<1x4x8x8xf32>
  %groups = tensor.expand_shape %padded [[0], [1, 2], [3], [4]] output_shape [1, 2, 2, 8, 8] : tensor<1x4x8x8xf32> into tensor<1x2x2x8x8xf32>
  %init = arith.constant dense<0.0> : tensor<1x2x3x6x6xf32>
  %conv = linalg.conv_2d_ngchw_gfchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%groups, %w : tensor<1x2x2x8x8xf32>, tensor<2x3x2x3x3xf32>) outs(%init : tensor<1x2x3x6x6xf32>) -> tensor<1x2x3x6x6xf32>
  %out = tensor.collapse_shape %conv [[0], [1, 2], [3], [4]] : tensor<1x2x3x6x6xf32> into tensor<1x6x6x6xf32>
  return %out : tensor<1x6x6x6xf32>
}
