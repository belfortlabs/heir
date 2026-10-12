module {
  func.func @conv2d_nchw_strided_pad(%arg0 : tensor<1x1x8x8xf32> {secret.secret}) -> tensor<1x4x4x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %filter1 = arith.constant dense<[[[[-1.0, 0.5], [-0.5, 1.0]]], [[[0.0, -1.0], [0.5, -0.5]]], [[[1.0, 0.0], [-1.0, 0.5]]], [[[-0.5, 1.0], [0.0, -1.0]]]]> : tensor<4x1x2x2xf32>
    %filter2 = arith.constant dense<[[[[-0.5, 1.0, 0.0], [-1.0, 0.5, -0.5], [1.0, 0.0, -1.0]], [[0.5, -0.5, 1.0], [0.0, -1.0, 0.5], [-0.5, 1.0, 0.0]], [[-1.0, 0.5, -0.5], [1.0, 0.0, -1.0], [0.5, -0.5, 1.0]], [[0.0, -1.0, 0.5], [-0.5, 1.0, 0.0], [-1.0, 0.5, -0.5]]], [[[1.0, 0.0, -1.0], [0.5, -0.5, 1.0], [0.0, -1.0, 0.5]], [[-0.5, 1.0, 0.0], [-1.0, 0.5, -0.5], [1.0, 0.0, -1.0]], [[0.5, -0.5, 1.0], [0.0, -1.0, 0.5], [-0.5, 1.0, 0.0]], [[-1.0, 0.5, -0.5], [1.0, 0.0, -1.0], [0.5, -0.5, 1.0]]], [[[0.0, -1.0, 0.5], [-0.5, 1.0, 0.0], [-1.0, 0.5, -0.5]], [[1.0, 0.0, -1.0], [0.5, -0.5, 1.0], [0.0, -1.0, 0.5]], [[-0.5, 1.0, 0.0], [-1.0, 0.5, -0.5], [1.0, 0.0, -1.0]], [[0.5, -0.5, 1.0], [0.0, -1.0, 0.5], [-0.5, 1.0, 0.0]]], [[[-1.0, 0.5, -0.5], [1.0, 0.0, -1.0], [0.5, -0.5, 1.0]], [[0.0, -1.0, 0.5], [-0.5, 1.0, 0.0], [-1.0, 0.5, -0.5]], [[1.0, 0.0, -1.0], [0.5, -0.5, 1.0], [0.0, -1.0, 0.5]], [[-0.5, 1.0, 0.0], [-1.0, 0.5, -0.5], [1.0, 0.0, -1.0]]]]> : tensor<4x4x3x3xf32>
    %zeros = arith.constant dense<0.000000e+00> : tensor<1x4x4x4xf32>
    %strided = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %filter1 : tensor<1x1x8x8xf32>, tensor<4x1x2x2xf32>) outs(%zeros : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    %padded = tensor.pad %strided low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%a: index, %b: index, %c: index, %d: index):
      tensor.yield %cst : f32
    } : tensor<1x4x4x4xf32> to tensor<1x4x6x6xf32>
    %out = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded, %filter2 : tensor<1x4x6x6xf32>, tensor<4x4x3x3xf32>) outs(%zeros : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    return %out : tensor<1x4x4x4xf32>
  }
}
