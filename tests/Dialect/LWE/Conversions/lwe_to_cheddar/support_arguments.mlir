// RUN: heir-opt --lwe-to-cheddar %s | FileCheck %s --implicit-check-not=!ctx --implicit-check-not=!eval_key
// RUN: heir-opt --lwe-to-cheddar=use-cyclops-runtime=true %s | FileCheck %s --implicit-check-not=!ctx --implicit-check-not=!eval_key

#encoding = #lwe.inverse_canonical_encoding<scaling_factor = 1099511627776>
#key = #lwe.key<>
#ring_f64 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64, encoding = #encoding>>

module attributes {
  scheme.ckks,
  ckks.schemeParam = #ckks.scheme_param<
    logN = 16,
    Q = [36028797019488257, 1099512938497],
    P = [2305843009211596801],
    logDefaultScale = 40
  >
} {
  // CHECK: func.func private @encode(%[[ENC:[a-zA-Z0-9_]+]]: !encoder,
  // CHECK: cheddar.encode %[[ENC]]
  func.func private @encode(%input: tensor<4xf64>) -> !pt {
    %0 = lwe.rlwe_encode %input {encoding = #encoding, ring = #ring_f64,
      level = 1 : i64, scale = 40 : i64} : tensor<4xf64> -> !pt
    return %0 : !pt
  }
  // CHECK: func.func private @forward(%[[ENC:[a-zA-Z0-9_]+]]: !encoder,
  // CHECK: call @encode(%[[ENC]],
  func.func private @forward(%input: tensor<4xf64>) -> !pt {
    %0 = call @encode(%input) : (tensor<4xf64>) -> !pt
    return %0 : !pt
  }
  // CHECK: func.func @entry(%[[ENC:[a-zA-Z0-9_]+]]: !encoder,
  // CHECK: call @forward(%[[ENC]],
  func.func @entry(%input: tensor<4xf64>) -> !pt {
    %0 = call @forward(%input) : (tensor<4xf64>) -> !pt
    return %0 : !pt
  }
}
