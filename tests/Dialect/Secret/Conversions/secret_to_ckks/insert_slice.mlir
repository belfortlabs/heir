// RUN: heir-opt --secret-to-ckks --canonicalize --split-input-file %s | FileCheck %s

// CHECK: ![[ct:.*]] = !lwe.lwe_ciphertext

#layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = i1 and (2i2 - i3 + slot) mod 4 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 1 and 0 <= i3 <= 1 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = tensor<1x2x2x2xf32>, layout = #layout>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  func.func @scalar(%arg0: !secret.secret<tensor<1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>, tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<1x1x4x4xf32>, layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and i1 = 0 and ct = 0 and (-4i2 - i3 + slot) mod 16 = 0 and 0 <= i2 <= 3 and 0 <= i3 <= 3 and 0 <= slot <= 1023 }">>}, %arg1: tensor<2x1x3x3xf32>) -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>, tensor_ext.original_type = #original_type}) {
    // CHECK: func.func @scalar
    // CHECK-SAME: (%[[arg0:.*]]: ![[ct]]
    // CHECK: %[[v0:.*]] = tensor.empty() : tensor<1x![[ct]]>
    // CHECK: %[[v1:.*]] = tensor.insert %[[arg0]] into %[[v0]]
    // CHECK: return %[[v1]] : tensor<1x![[ct]]>
    %3 = tensor.empty() : tensor<1x1024xf32>
    %5 = mgmt.init %3 {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>} : tensor<1x1024xf32>
    %11 = secret.generic(%arg0: !secret.secret<tensor<1024xf32>>) {
    ^body(%input0: tensor<1024xf32>):
      %inserted_slice_7 = tensor.insert_slice %input0 into %5[0, 0] [1, 1024] [1, 1] : tensor<1024xf32> into tensor<1x1024xf32>
      secret.yield %inserted_slice_7 : tensor<1x1024xf32>
    } -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>})
    return %11 : !secret.secret<tensor<1x1024xf32>>
  }
}

// -----

// CHECK: ![[ct:.*]] = !lwe.lwe_ciphertext

#layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = i1 and (2i2 - i3 + slot) mod 4 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 1 and 0 <= i3 <= 1 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = tensor<1x2x2x2xf32>, layout = #layout>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  func.func @tensor(%arg0: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 90>, tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<1x1x4x4xf32>, layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and i1 = 0 and ct = 0 and (-4i2 - i3 + slot) mod 16 = 0 and 0 <= i2 <= 3 and 0 <= i3 <= 3 and 0 <= slot <= 1023 }">>}, %arg1: tensor<2x1x3x3xf32>) -> (!secret.secret<tensor<2x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 90>, tensor_ext.original_type = #original_type}) {
    // CHECK: func.func @tensor
    // CHECK-SAME: (%[[arg0:.*]]: tensor<1x![[ct]]>
    // CHECK: %[[v0:.*]] = tensor.empty() : tensor<2x![[ct]]>
    // CHECK: %[[v1:.*]] = tensor.insert_slice %[[arg0]] into %[[v0]][1] [1] [1] : tensor<1x![[ct]]> into tensor<2x![[ct]]>
    // CHECK: return %[[v1]] : tensor<2x![[ct]]>
    %4 = tensor.empty() : tensor<2x1024xf32>
    %9 = mgmt.init %4 {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 90>} : tensor<2x1024xf32>
    %46 = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %inserted_slice_7 = tensor.insert_slice %input0 into %9[1, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> into tensor<2x1024xf32>
      secret.yield %inserted_slice_7 : tensor<2x1024xf32>
    } -> (!secret.secret<tensor<2x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 90>})
    return %46 : !secret.secret<tensor<2x1024xf32>>
  }
}

// -----

// A cleartext row inserted into a tensor of ciphertexts is trivially
// encrypted.

// CHECK: ![[ct:.*]] = !lwe.lwe_ciphertext

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  // CHECK: func.func @cleartext_row
  // CHECK-SAME: (%[[arg0:.*]]: tensor<2x![[ct]]>
  // CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  // CHECK: %[[pt:.*]] = lwe.rlwe_encode %[[cst]]
  // CHECK: %[[pts:.*]] = tensor.from_elements %[[pt]] : tensor<1x
  // CHECK: %[[row:.*]] = lwe.trivial_encrypt %[[pts]] : tensor<1x!{{.*}}> -> tensor<1x![[ct]]>
  // CHECK: %[[v1:.*]] = tensor.insert_slice %[[row]] into %[[arg0]][1] [1] [1] : tensor<1x![[ct]]> into tensor<2x![[ct]]>
  // CHECK: return %[[v1]] : tensor<2x![[ct]]>
  func.func @cleartext_row(%arg0: !secret.secret<tensor<2x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>}) -> (!secret.secret<tensor<2x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>}) {
    %cst = arith.constant dense<0.0> : tensor<1x1024xf32>
    %0 = mgmt.init %cst {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>} : tensor<1x1024xf32>
    %1 = secret.generic(%arg0: !secret.secret<tensor<2x1024xf32>>) {
    ^body(%input0: tensor<2x1024xf32>):
      %inserted_slice = tensor.insert_slice %0 into %input0[1, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> into tensor<2x1024xf32>
      secret.yield %inserted_slice : tensor<2x1024xf32>
    } -> (!secret.secret<tensor<2x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>})
    return %1 : !secret.secret<tensor<2x1024xf32>>
  }
}

// -----

// A ciphertext inserted into a cleartext tensor makes the rows of that tensor
// trivial encryptions.

// CHECK: ![[ct:.*]] = !lwe.lwe_ciphertext

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  // CHECK: func.func @cleartext_dest
  // CHECK-SAME: (%[[arg0:.*]]: tensor<1x![[ct]]>
  // CHECK: %[[pts:.*]] = tensor.from_elements {{.*}} : tensor<2x
  // CHECK: %[[dest:.*]] = lwe.trivial_encrypt %[[pts]] : tensor<2x!{{.*}}> -> tensor<2x![[ct]]>
  // CHECK: %[[v1:.*]] = tensor.insert_slice %[[arg0]] into %[[dest]][1] [1] [1] : tensor<1x![[ct]]> into tensor<2x![[ct]]>
  // CHECK: return %[[v1]] : tensor<2x![[ct]]>
  func.func @cleartext_dest(%arg0: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>}) -> (!secret.secret<tensor<2x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>}) {
    %cst = arith.constant dense<0.0> : tensor<2x1024xf32>
    %0 = mgmt.init %cst {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>} : tensor<2x1024xf32>
    %1 = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %inserted_slice = tensor.insert_slice %input0 into %0[1, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> into tensor<2x1024xf32>
      secret.yield %inserted_slice : tensor<2x1024xf32>
    } -> (!secret.secret<tensor<2x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>})
    return %1 : !secret.secret<tensor<2x1024xf32>>
  }
}
