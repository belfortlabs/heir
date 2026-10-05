// RUN: heir-opt --layout-propagation=min-slot-count=64 --split-input-file --verify-diagnostics %s

module attributes {backend.openfhe} {
  func.func @no_kernel(%arg0: !secret.secret<tensor<1x4x8xf32>>) -> !secret.secret<tensor<1x4x4xf32>> {
    %neg_inf = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<1x4x4xf32>
    %1 = linalg.fill ins(%neg_inf : f32) outs(%0 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    %window = tensor.empty() : tensor<2xf32>
    %r = secret.generic(%arg0 : !secret.secret<tensor<1x4x8xf32>>) {
    ^body(%x: tensor<1x4x8xf32>):
      // expected-error@below {{requires a backend with a max pool kernel}}
      %2 = linalg.pooling_ncw_max {dilations = dense<1> : vector<1xi64>, domain_lower = 0.0 : f64, domain_upper = 1.0 : f64, strides = dense<2> : vector<1xi64>} ins(%x, %window : tensor<1x4x8xf32>, tensor<2xf32>) outs(%1 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
      secret.yield %2 : tensor<1x4x4xf32>
    } -> !secret.secret<tensor<1x4x4xf32>>
    return %r : !secret.secret<tensor<1x4x4xf32>>
  }
}

// -----

module attributes {backend.cheddar} {
  func.func @no_domain(%arg0: !secret.secret<tensor<1x4x8xf32>>) -> !secret.secret<tensor<1x4x4xf32>> {
    %neg_inf = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<1x4x4xf32>
    %1 = linalg.fill ins(%neg_inf : f32) outs(%0 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    %window = tensor.empty() : tensor<2xf32>
    %r = secret.generic(%arg0 : !secret.secret<tensor<1x4x8xf32>>) {
    ^body(%x: tensor<1x4x8xf32>):
      // expected-error@below {{requires float domain_lower/domain_upper attributes}}
      %2 = linalg.pooling_ncw_max {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>} ins(%x, %window : tensor<1x4x8xf32>, tensor<2xf32>) outs(%1 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
      secret.yield %2 : tensor<1x4x4xf32>
    } -> !secret.secret<tensor<1x4x4xf32>>
    return %r : !secret.secret<tensor<1x4x4xf32>>
  }
}

// -----

module attributes {backend.cheddar} {
  func.func @untrimmed_tail(%arg0: !secret.secret<tensor<1x4x9xf32>>) -> !secret.secret<tensor<1x4x4xf32>> {
    %neg_inf = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<1x4x4xf32>
    %1 = linalg.fill ins(%neg_inf : f32) outs(%0 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    %window = tensor.empty() : tensor<2xf32>
    %r = secret.generic(%arg0 : !secret.secret<tensor<1x4x9xf32>>) {
    ^body(%x: tensor<1x4x9xf32>):
      // expected-error@below {{requires an input length that the windows tile exactly}}
      %2 = linalg.pooling_ncw_max {dilations = dense<1> : vector<1xi64>, domain_lower = 0.0 : f64, domain_upper = 1.0 : f64, strides = dense<2> : vector<1xi64>} ins(%x, %window : tensor<1x4x9xf32>, tensor<2xf32>) outs(%1 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
      secret.yield %2 : tensor<1x4x4xf32>
    } -> !secret.secret<tensor<1x4x4xf32>>
    return %r : !secret.secret<tensor<1x4x4xf32>>
  }
}

// -----

module attributes {backend.cheddar} {
  func.func @too_many_slots(%arg0: !secret.secret<tensor<1x16x8xf32>>) -> !secret.secret<tensor<1x16x4xf32>> {
    %neg_inf = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<1x16x4xf32>
    %1 = linalg.fill ins(%neg_inf : f32) outs(%0 : tensor<1x16x4xf32>) -> tensor<1x16x4xf32>
    %window = tensor.empty() : tensor<2xf32>
    %r = secret.generic(%arg0 : !secret.secret<tensor<1x16x8xf32>>) {
    ^body(%x: tensor<1x16x8xf32>):
      // expected-error@below {{requires the input (128 elements) to fit one ciphertext of 64 slots}}
      %2 = linalg.pooling_ncw_max {dilations = dense<1> : vector<1xi64>, domain_lower = 0.0 : f64, domain_upper = 1.0 : f64, strides = dense<2> : vector<1xi64>} ins(%x, %window : tensor<1x16x8xf32>, tensor<2xf32>) outs(%1 : tensor<1x16x4xf32>) -> tensor<1x16x4xf32>
      secret.yield %2 : tensor<1x16x4xf32>
    } -> !secret.secret<tensor<1x16x4xf32>>
    return %r : !secret.secret<tensor<1x16x4xf32>>
  }
}
