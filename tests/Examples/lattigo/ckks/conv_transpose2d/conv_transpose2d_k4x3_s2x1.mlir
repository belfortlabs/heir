#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @conv_transpose2d_k4x3_s2x1(%arg0: tensor<1x2x4x4xf32> {secret.secret}) -> tensor<1x3x9x6xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<[[[[-0.625, -0.375, -0.125], [0.125, 0.375, 0.625], [-0.5, -0.25, 0.0], [0.25, 0.5, -0.625]], [[-0.375, -0.125, 0.125], [0.375, 0.625, -0.5], [-0.25, 0.0, 0.25], [0.5, -0.625, -0.375]], [[-0.125, 0.125, 0.375], [0.625, -0.5, -0.25], [0.0, 0.25, 0.5], [-0.625, -0.375, -0.125]]], [[[0.125, 0.375, 0.625], [-0.5, -0.25, 0.0], [0.25, 0.5, -0.625], [-0.375, -0.125, 0.125]], [[0.375, 0.625, -0.5], [-0.25, 0.0, 0.25], [0.5, -0.625, -0.375], [-0.125, 0.125, 0.375]], [[0.625, -0.5, -0.25], [0.0, 0.25, 0.5], [-0.625, -0.375, -0.125], [0.125, 0.375, 0.625]]]]> : tensor<2x3x4x3xf32>
    %cst_1 = arith.constant dense<[-0.5, 0.0, 0.5]> : tensor<3xf32>
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %0 = tensor.empty() : tensor<3x2x4x3xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<3x2x4x3xf32>) -> tensor<3x2x4x3xf32>
    %2 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1 : tensor<3x2x4x3xf32>) {
    ^bb0(%out: f32):
      %7 = linalg.index 0 : index
      %8 = linalg.index 1 : index
      %9 = linalg.index 2 : index
      %10 = linalg.index 3 : index
      %11 = arith.subi %c3, %9 : index
      %12 = arith.subi %c2, %10 : index
      %extracted = tensor.extract %cst_0[%8, %7, %11, %12] : tensor<2x3x4x3xf32>
      linalg.yield %extracted : f32
    } -> tensor<3x2x4x3xf32>
    %3 = tensor.empty() : tensor<1x2x12x8xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1x2x12x8xf32>) -> tensor<1x2x12x8xf32>
    %inserted_slice = tensor.insert_slice %arg0 into %4[0, 0, 2, 2] [1, 2, 4, 4] [1, 1, 2, 1] : tensor<1x2x4x4xf32> into tensor<1x2x12x8xf32>
    %5 = tensor.empty() : tensor<1x3x9x6xf32>
    %broadcasted = linalg.broadcast ins(%cst_1 : tensor<3xf32>) outs(%5 : tensor<1x3x9x6xf32>) dimensions = [0, 2, 3]
    %6 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%inserted_slice, %2 : tensor<1x2x12x8xf32>, tensor<3x2x4x3xf32>) outs(%broadcasted : tensor<1x3x9x6xf32>) -> tensor<1x3x9x6xf32>
    return %6 : tensor<1x3x9x6xf32>
  }
}
