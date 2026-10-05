// A ceil-mode MaxPool1d(8) over 4 channels of 61 values in [-3, 5], in the
// form that torch-mlir gives: a -inf pad to 64 values, then the pool. The
// windows of 8 slots keep the levels free of Cyclops' small-window league
// level cap (requirement R2 in cyclops_maxpool_requirements.md).
func.func @max_pool(%arg0: tensor<1x4x61xf32> {secret.secret}) -> tensor<1x4x8xf32> {
  %neg_inf = arith.constant 0xFF800000 : f32
  %padded = tensor.pad %arg0 low[0, 0, 0] high[0, 0, 3] {
  ^bb0(%i: index, %j: index, %k: index):
    tensor.yield %neg_inf : f32
  } : tensor<1x4x61xf32> to tensor<1x4x64xf32>
  %0 = tensor.empty() : tensor<1x4x8xf32>
  %1 = linalg.fill ins(%neg_inf : f32) outs(%0 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
  %window = tensor.empty() : tensor<8xf32>
  %2 = linalg.pooling_ncw_max {dilations = dense<1> : vector<1xi64>, domain_lower = -3.0 : f64, domain_upper = 5.0 : f64, strides = dense<8> : vector<1xi64>} ins(%padded, %window : tensor<1x4x64xf32>, tensor<8xf32>) outs(%1 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
  return %2 : tensor<1x4x8xf32>
}
