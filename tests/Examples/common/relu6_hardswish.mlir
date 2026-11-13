#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
module {
  // relu6(x) + hardswish(x) in the shapes torch-mlir exports them, with the
  // calibrated domains on the generics:
  //   - relu6(x) = min(6, max(x, 0)), with the bounds as 0-d tensors compared
  //     with ordered predicates,
  //   - hardswish(x) = x * min(relu(x + 3), 6) / 6, with a ugt ReLU and a
  //     division by a scalar constant.
  // activation-canonicalizations turns the compare/selects into maximumf and
  // minimumf, linalg-canonicalizations turns the division into a
  // multiplication, and polynomial-approximation splits each clamp into two
  // composite ReLUs.
  func.func @relu6_hardswish(%arg0: tensor<1x16xf32> {secret.secret}) -> tensor<1x16xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_6 = arith.constant dense<6.000000e+00> : tensor<f32>
    %c3 = arith.constant 3.000000e+00 : f32
    %c6 = arith.constant 6.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x16xf32>
    // relu6
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"], domain_lower = -7.5 : f64, domain_upper = 7.5 : f64} ins(%arg0, %cst_0 : tensor<1x16xf32>, tensor<f32>) outs(%0 : tensor<1x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %10 = arith.cmpf ogt, %in, %in_0 : f32
      %11 = arith.select %10, %in, %in_0 : f32
      linalg.yield %11 : f32
    } -> tensor<1x16xf32>
    %2 = linalg.generic {indexing_maps = [#map1, #map, #map], iterator_types = ["parallel", "parallel"], domain_lower = 0.0 : f64, domain_upper = 7.5 : f64} ins(%cst_6, %1 : tensor<f32>, tensor<1x16xf32>) outs(%0 : tensor<1x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %10 = arith.cmpf olt, %in, %in_0 : f32
      %11 = arith.select %10, %in, %in_0 : f32
      linalg.yield %11 : f32
    } -> tensor<1x16xf32>
    // hardswish
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x16xf32>) outs(%0 : tensor<1x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %10 = arith.addf %in, %c3 : f32
      linalg.yield %10 : f32
    } -> tensor<1x16xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"], domain_lower = -4.5 : f64, domain_upper = 10.5 : f64} ins(%3 : tensor<1x16xf32>) outs(%0 : tensor<1x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %10 = arith.cmpf ugt, %in, %cst : f32
      %11 = arith.select %10, %in, %cst : f32
      linalg.yield %11 : f32
    } -> tensor<1x16xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"], domain_lower = 0.0 : f64, domain_upper = 10.5 : f64} ins(%4, %cst_6 : tensor<1x16xf32>, tensor<f32>) outs(%0 : tensor<1x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %10 = arith.cmpf olt, %in, %in_0 : f32
      %11 = arith.select %10, %in, %in_0 : f32
      linalg.yield %11 : f32
    } -> tensor<1x16xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<1x16xf32>) outs(%0 : tensor<1x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %10 = arith.divf %in, %c6 : f32
      linalg.yield %10 : f32
    } -> tensor<1x16xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%6, %arg0 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%0 : tensor<1x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %10 = arith.mulf %in, %in_0 : f32
      linalg.yield %10 : f32
    } -> tensor<1x16xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %7 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%0 : tensor<1x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %10 = arith.addf %in, %in_0 : f32
      linalg.yield %10 : f32
    } -> tensor<1x16xf32>
    return %8 : tensor<1x16xf32>
  }
}
