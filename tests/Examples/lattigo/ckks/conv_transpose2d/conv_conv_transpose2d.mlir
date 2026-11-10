#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @conv_conv_transpose2d(%arg0: tensor<1x2x4x4xf32> {secret.secret}) -> tensor<1x3x8x8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<[0.625, 0.125, -0.375]> : tensor<3xf32>
    %cst_1 = arith.constant dense<[[[[0.5, -0.25, 0.375], [-0.375, 0.25, -0.5], [0.125, -0.625, 0.0]], [[0.625, -0.125, 0.5], [-0.25, 0.375, -0.375], [0.25, -0.5, 0.125]], [[-0.625, 0.0, 0.625], [-0.125, 0.5, -0.25], [0.375, -0.375, 0.25]]], [[[-0.5, 0.125, -0.625], [0.0, 0.625, -0.125], [0.5, -0.25, 0.375]], [[-0.375, 0.25, -0.5], [0.125, -0.625, 0.0], [0.625, -0.125, 0.5]], [[-0.25, 0.375, -0.375], [0.25, -0.5, 0.125], [-0.625, 0.0, 0.625]]]]> : tensor<2x3x3x3xf32>
    %cst_2 = arith.constant dense<[[[[0.25, 0.5, -0.625], [-0.375, -0.125, 0.125], [0.375, 0.625, -0.5]], [[-0.25, 0.0, 0.25], [0.5, -0.625, -0.375], [-0.125, 0.125, 0.375]]], [[[0.625, -0.5, -0.25], [0.0, 0.25, 0.5], [-0.625, -0.375, -0.125]], [[0.125, 0.375, 0.625], [-0.5, -0.25, 0.0], [0.25, 0.5, -0.625]]]]> : tensor<2x2x3x3xf32>
    %cst_3 = arith.constant dense<[0.375, -0.5]> : tensor<2xf32>
    %c2 = arith.constant 2 : index
    %padded = tensor.pad %arg0 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x2x4x4xf32> to tensor<1x2x6x6xf32>
    %0 = tensor.empty() : tensor<1x2x4x4xf32>
    %broadcasted = linalg.broadcast ins(%cst_3 : tensor<2xf32>) outs(%0 : tensor<1x2x4x4xf32>) dimensions = [0, 2, 3]
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded, %cst_2 : tensor<1x2x6x6xf32>, tensor<2x2x3x3xf32>) outs(%broadcasted : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    %2 = tensor.empty() : tensor<3x2x3x3xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<3x2x3x3xf32>) -> tensor<3x2x3x3xf32>
    %4 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3 : tensor<3x2x3x3xf32>) {
    ^bb0(%out: f32):
      %9 = linalg.index 0 : index
      %10 = linalg.index 1 : index
      %11 = linalg.index 2 : index
      %12 = linalg.index 3 : index
      %13 = arith.subi %c2, %11 : index
      %14 = arith.subi %c2, %12 : index
      %extracted = tensor.extract %cst_1[%10, %9, %13, %14] : tensor<2x3x3x3xf32>
      linalg.yield %extracted : f32
    } -> tensor<3x2x3x3xf32>
    %5 = tensor.empty() : tensor<1x2x10x10xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x2x10x10xf32>) -> tensor<1x2x10x10xf32>
    %inserted_slice = tensor.insert_slice %1 into %6[0, 0, 1, 1] [1, 2, 4, 4] [1, 1, 2, 2] : tensor<1x2x4x4xf32> into tensor<1x2x10x10xf32>
    %7 = tensor.empty() : tensor<1x3x8x8xf32>
    %broadcasted_4 = linalg.broadcast ins(%cst_0 : tensor<3xf32>) outs(%7 : tensor<1x3x8x8xf32>) dimensions = [0, 2, 3]
    %8 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%inserted_slice, %4 : tensor<1x2x10x10xf32>, tensor<3x2x3x3xf32>) outs(%broadcasted_4 : tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
    return %8 : tensor<1x3x8x8xf32>
  }
}
