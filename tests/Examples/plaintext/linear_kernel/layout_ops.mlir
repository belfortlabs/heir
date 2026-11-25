// Operations that rename, select or copy elements of a secret tensor.

// A transpose and the flattening behind it are free. Taking every fourth
// element of the flat vector is not: those elements are not at regular
// positions, so the slice moves them.
func.func @transpose_matvec(%x: tensor<2x3x4xf32> {secret.secret}, %m: tensor<5x6xf32>) -> tensor<5xf32> {
  %empty = tensor.empty() : tensor<4x2x3xf32>
  %t = linalg.transpose ins(%x : tensor<2x3x4xf32>) outs(%empty : tensor<4x2x3xf32>) permutation = [2, 0, 1]
  %flat = tensor.collapse_shape %t [[0, 1, 2]] : tensor<4x2x3xf32> into tensor<24xf32>
  %s = tensor.extract_slice %flat[0] [6] [4] : tensor<24xf32> to tensor<6xf32>
  %init = arith.constant dense<0.0> : tensor<5xf32>
  %out = linalg.matvec ins(%m, %s : tensor<5x6xf32>, tensor<6xf32>) outs(%init : tensor<5xf32>) -> tensor<5xf32>
  return %out : tensor<5xf32>
}

// Every second row, starting at row 1, with the unit axis dropped.
func.func @slice_rows(%x: tensor<1x8x8xf32> {secret.secret}) -> tensor<4x8xf32> {
  %s = tensor.extract_slice %x[0, 1, 0] [1, 4, 8] [1, 2, 1] : tensor<1x8x8xf32> to tensor<4x8xf32>
  return %s : tensor<4x8xf32>
}

// A secret vector broadcast along a new axis and added to a secret matrix.
func.func @broadcast_add(%x: tensor<4xf32> {secret.secret}, %y: tensor<4x8xf32> {secret.secret}) -> tensor<4x8xf32> {
  %empty = tensor.empty() : tensor<4x8xf32>
  %b = linalg.broadcast ins(%x : tensor<4xf32>) outs(%empty : tensor<4x8xf32>) dimensions = [1]
  %out = arith.addf %b, %y : tensor<4x8xf32>
  return %out : tensor<4x8xf32>
}

// Two secret tensors concatenated along the channel axis.
func.func @concat(%x: tensor<1x2x4x4xf32> {secret.secret}, %y: tensor<1x3x4x4xf32> {secret.secret}) -> tensor<1x5x4x4xf32> {
  %out = tensor.concat dim(1) %x, %y : (tensor<1x2x4x4xf32>, tensor<1x3x4x4xf32>) -> tensor<1x5x4x4xf32>
  return %out : tensor<1x5x4x4xf32>
}

// A batch normalization whose statistics are cleartext arguments, between a
// strided convolution, which leaves gaps in the layout, and a second
// convolution, which reads across those gaps. It is written in the form the
// preprocessing passes leave it in: the mean is the initial value of the
// first convolution, computed in cleartext, and the scale is a broadcast
// cleartext factor.
func.func @conv_clear_init(%x: tensor<1x2x8x8xf32> {secret.secret}, %mean: tensor<3xf32>, %scale: tensor<3xf32>, %w1: tensor<3x2x2x2xf32>, %w2: tensor<2x3x3x3xf32>) -> tensor<1x2x4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %zeros = arith.constant dense<0.0> : tensor<1x3x4x4xf32>
  %e1 = tensor.empty() : tensor<1x3x4x4xf32>
  %bm = linalg.broadcast ins(%mean : tensor<3xf32>) outs(%e1 : tensor<1x3x4x4xf32>) dimensions = [0, 2, 3]
  // The scale is computed in cleartext, as a batch normalization computes it
  // from the variance.
  %half = arith.constant dense<0.5> : tensor<3xf32>
  %twice = arith.addf %scale, %scale : tensor<3xf32>
  %computed = arith.mulf %twice, %half : tensor<3xf32>
  %bs = linalg.broadcast ins(%computed : tensor<3xf32>) outs(%e1 : tensor<1x3x4x4xf32>) dimensions = [0, 2, 3]
  %init = arith.subf %zeros, %bm : tensor<1x3x4x4xf32>
  %c1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%x, %w1 : tensor<1x2x8x8xf32>, tensor<3x2x2x2xf32>) outs(%init : tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
  %mul = arith.mulf %c1, %bs : tensor<1x3x4x4xf32>
  %padded = tensor.pad %mul low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %zero : f32
  } : tensor<1x3x4x4xf32> to tensor<1x3x6x6xf32>
  %e2 = tensor.empty() : tensor<1x2x4x4xf32>
  %f2 = linalg.fill ins(%zero : f32) outs(%e2 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %c2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded, %w2 : tensor<1x3x6x6xf32>, tensor<2x3x3x3xf32>) outs(%f2 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  return %c2 : tensor<1x2x4x4xf32>
}
