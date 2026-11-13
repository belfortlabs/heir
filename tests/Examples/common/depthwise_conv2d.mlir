module {
  func.func @depthwise_conv2d(%arg0 : tensor<1x4x8x8xf32> {secret.secret}) -> tensor<1x2x8x8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %dw_filter = arith.constant dense<[[[[1.0, 0.0, -1.0], [0.5, 1.0, 0.5], [-1.0, 0.0, 1.0]]], [[[0.0, 0.5, 0.0], [0.5, -1.0, 0.5], [0.0, 0.5, 0.0]]], [[[-0.5, 1.0, -0.5], [1.0, 0.0, -1.0], [0.5, -1.0, 0.5]]], [[[0.25, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.25]]]]> : tensor<4x1x3x3xf32>
    %dw_bias = arith.constant dense<[0.1, -0.1, 0.2, 0.0]> : tensor<4xf32>
    %pw_filter = arith.constant dense<[[[[0.5]], [[-1.0]], [[1.0]], [[0.25]]], [[[-0.5]], [[0.5]], [[1.0]], [[-1.0]]]]> : tensor<2x4x1x1xf32>
    %pw_bias = arith.constant dense<[0.0, 0.1]> : tensor<2xf32>
    %padded = tensor.pad %arg0 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%a: index, %b: index, %c: index, %d: index):
      tensor.yield %cst : f32
    } : tensor<1x4x8x8xf32> to tensor<1x4x10x10xf32>
    %collapsed = tensor.collapse_shape %dw_filter [[0, 1], [2], [3]] : tensor<4x1x3x3xf32> into tensor<4x3x3xf32>
    %empty0 = tensor.empty() : tensor<1x4x8x8xf32>
    %bcast0 = linalg.broadcast ins(%dw_bias : tensor<4xf32>) outs(%empty0 : tensor<1x4x8x8xf32>) dimensions = [0, 2, 3]
    %dw = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded, %collapsed : tensor<1x4x10x10xf32>, tensor<4x3x3xf32>) outs(%bcast0 : tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32>
    %empty1 = tensor.empty() : tensor<1x2x8x8xf32>
    %bcast1 = linalg.broadcast ins(%pw_bias : tensor<2xf32>) outs(%empty1 : tensor<1x2x8x8xf32>) dimensions = [0, 2, 3]
    %out = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%dw, %pw_filter : tensor<1x4x8x8xf32>, tensor<2x4x1x1xf32>) outs(%bcast1 : tensor<1x2x8x8xf32>) -> tensor<1x2x8x8xf32>
    return %out : tensor<1x2x8x8xf32>
  }
}
