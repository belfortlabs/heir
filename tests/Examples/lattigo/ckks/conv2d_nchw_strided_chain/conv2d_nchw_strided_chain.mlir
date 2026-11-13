// A stride-2 conv on the pixel-shuffled result of another stride-2 conv, with
// 5 output channels, which is not a multiple of gap^2 = 16. The second conv
// reads its data in the shuffled layout, so its result is shuffled by a gap of
// 2 * 2 = 4 and reserves one block of 16 channels.
module {
  func.func @conv2d_strided_chain(%arg0: tensor<1x1x8x8xf32> {secret.secret}) -> tensor<1x5x2x2xf32> {
    %filter1 = arith.constant dense<[[[[-0.75, -0.5], [-0.25, 0.0]]], [[[0.0, 0.25], [0.5, 0.75]]], [[[0.75, -0.75], [-0.5, -0.25]]], [[[-0.25, 0.0], [0.25, 0.5]]]]> : tensor<4x1x2x2xf32>
    %filter2 = arith.constant dense<[[[[-0.5, -0.125], [-0.25, 0.125]], [[-0.125, 0.25], [0.125, 0.5]], [[0.25, -0.5], [0.5, -0.25]], [[-0.5, -0.125], [-0.25, 0.125]]], [[[0.125, 0.5], [0.375, -0.375]], [[0.5, -0.25], [-0.375, 0.0]], [[-0.25, 0.125], [0.0, 0.375]], [[0.125, 0.5], [0.375, -0.375]]], [[[-0.375, 0.0], [-0.125, 0.25]], [[0.0, 0.375], [0.25, -0.5]], [[0.375, -0.375], [-0.5, -0.125]], [[-0.375, 0.0], [-0.125, 0.25]]], [[[0.25, -0.5], [0.5, -0.25]], [[-0.5, -0.125], [-0.25, 0.125]], [[-0.125, 0.25], [0.125, 0.5]], [[0.25, -0.5], [0.5, -0.25]]], [[[-0.25, 0.125], [0.0, 0.375]], [[0.125, 0.5], [0.375, -0.375]], [[0.5, -0.25], [-0.375, 0.0]], [[-0.25, 0.125], [0.0, 0.375]]]]> : tensor<5x4x2x2xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %e1 = tensor.empty() : tensor<1x4x4x4xf32>
    %init1 = linalg.fill ins(%cst : f32) outs(%e1 : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    %conv1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%arg0, %filter1 : tensor<1x1x8x8xf32>, tensor<4x1x2x2xf32>)
      outs(%init1 : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    %e2 = tensor.empty() : tensor<1x5x2x2xf32>
    %init2 = linalg.fill ins(%cst : f32) outs(%e2 : tensor<1x5x2x2xf32>) -> tensor<1x5x2x2xf32>
    %conv2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%conv1, %filter2 : tensor<1x4x4x4xf32>, tensor<5x4x2x2xf32>)
      outs(%init2 : tensor<1x5x2x2xf32>) -> tensor<1x5x2x2xf32>
    return %conv2 : tensor<1x5x2x2xf32>
  }
}
