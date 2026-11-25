// A residual block with a stride-2 convolution and a 1x1 stride-2 shortcut,
// a squaring activation, a cleartext scale, a flatten and a dense layer; and
// a sum of two secret tensors whose layouts differ.

func.func @resblock(%x: tensor<1x2x8x8xf32> {secret.secret}, %w1: tensor<4x2x3x3xf32>, %w2: tensor<4x4x3x3xf32>, %ws: tensor<4x4x1x1xf32>, %m: tensor<10x64xf32>) -> tensor<10xf32> {
  %zero = arith.constant 0.0 : f32
  %init1 = arith.constant dense<0.0> : tensor<1x4x8x8xf32>
  %init2 = arith.constant dense<0.0> : tensor<1x4x4x4xf32>
  %init3 = arith.constant dense<0.0> : tensor<10xf32>
  %half = arith.constant dense<0.5> : tensor<1x4x4x4xf32>

  %padded1 = tensor.pad %x low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %zero : f32
  } : tensor<1x2x8x8xf32> to tensor<1x2x10x10xf32>
  %h1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded1, %w1 : tensor<1x2x10x10xf32>, tensor<4x2x3x3xf32>) outs(%init1 : tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32>
  %a1 = arith.mulf %h1, %h1 : tensor<1x4x8x8xf32>

  %padded2 = tensor.pad %a1 low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %zero : f32
  } : tensor<1x4x8x8xf32> to tensor<1x4x10x10xf32>
  %h2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded2, %w2 : tensor<1x4x10x10xf32>, tensor<4x4x3x3xf32>) outs(%init2 : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  %shortcut = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%a1, %ws : tensor<1x4x8x8xf32>, tensor<4x4x1x1xf32>) outs(%init2 : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  %sum = arith.addf %h2, %shortcut : tensor<1x4x4x4xf32>
  %scaled = arith.mulf %sum, %half : tensor<1x4x4x4xf32>

  %flat = tensor.collapse_shape %scaled [[0, 1, 2, 3]] : tensor<1x4x4x4xf32> into tensor<64xf32>
  %out = linalg.matvec ins(%m, %flat : tensor<10x64xf32>, tensor<64xf32>) outs(%init3 : tensor<10xf32>) -> tensor<10xf32>
  return %out : tensor<10xf32>
}

func.func @join(%x: tensor<1x2x8x8xf32> {secret.secret}, %y: tensor<1x4x4x4xf32> {secret.secret}, %w: tensor<4x2x2x2xf32>) -> tensor<1x4x4x4xf32> {
  %init = arith.constant dense<0.0> : tensor<1x4x4x4xf32>
  %h = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%x, %w : tensor<1x2x8x8xf32>, tensor<4x2x2x2xf32>) outs(%init : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  %out = arith.addf %h, %y : tensor<1x4x4x4xf32>
  return %out : tensor<1x4x4x4xf32>
}
