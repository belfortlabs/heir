// A stride-1 3x3 conv on the zero-padded, pixel-shuffled result of a stride-2
// conv. The pad cannot fold into the conv for shuffled data, so the padded data
// is converted to the shuffled layout of its padded shape, and the conv reads it
// there. Its result keeps the gap of 2. The 3 channels of the first conv leave
// one channel of the 2x2 block empty.
module {
  func.func @conv2d_gapped_pad(%arg0: tensor<1x1x8x8xf32> {secret.secret}) -> tensor<1x2x4x4xf32> {
    %filter1 = arith.constant dense<[[[[-0.75, -0.5], [-0.25, 0.0]]], [[[0.0, 0.25], [0.5, 0.75]]], [[[0.75, -0.75], [-0.5, -0.25]]]]> : tensor<3x1x2x2xf32>
    %filter2 = arith.constant dense<[[[[-0.5, -0.125, 0.25], [-0.25, 0.125, 0.5], [0.0, 0.375, -0.375]], [[-0.125, 0.25, -0.5], [0.125, 0.5, -0.25], [0.375, -0.375, 0.0]], [[0.25, -0.5, -0.125], [0.5, -0.25, 0.125], [-0.375, 0.0, 0.375]]], [[[0.125, 0.5, -0.25], [0.375, -0.375, 0.0], [-0.5, -0.125, 0.25]], [[0.5, -0.25, 0.125], [-0.375, 0.0, 0.375], [-0.125, 0.25, -0.5]], [[-0.25, 0.125, 0.5], [0.0, 0.375, -0.375], [0.25, -0.5, -0.125]]]]> : tensor<2x3x3x3xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %e1 = tensor.empty() : tensor<1x3x4x4xf32>
    %init1 = linalg.fill ins(%cst : f32) outs(%e1 : tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    %conv1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%arg0, %filter1 : tensor<1x1x8x8xf32>, tensor<3x1x2x2xf32>)
      outs(%init1 : tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    %padded = tensor.pad %conv1 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
      tensor.yield %cst : f32
    } : tensor<1x3x4x4xf32> to tensor<1x3x6x6xf32>
    %e2 = tensor.empty() : tensor<1x2x4x4xf32>
    %init2 = linalg.fill ins(%cst : f32) outs(%e2 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    %conv2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%padded, %filter2 : tensor<1x3x6x6xf32>, tensor<2x3x3x3xf32>)
      outs(%init2 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    return %conv2 : tensor<1x2x4x4xf32>
  }
}
