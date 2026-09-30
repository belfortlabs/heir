// RUN: heir-opt --annotate-mgmt %s | FileCheck %s

// `mgmt.top_level` pins the level fresh ciphertexts sit at: the program is
// two levels deep but is placed at the top of a six-level chain.

// CHECK: module attributes {mgmt.top_level = 5 : i64}
module attributes {mgmt.top_level = 5 : i64} {
  // CHECK: func.func @main(%[[ARG:.*]]: !secret.secret<tensor<8xi8>> {mgmt.mgmt = #mgmt.mgmt<level = 5>})
  func.func @main(%arg0: !secret.secret<tensor<8xi8>>) -> !secret.secret<tensor<8xi8>> {
    // CHECK: mgmt.modreduce
    // CHECK-SAME: mgmt.mgmt = #mgmt.mgmt<level = 4>
    // CHECK: mgmt.modreduce
    // CHECK-SAME: mgmt.mgmt = #mgmt.mgmt<level = 3>
    %b = secret.generic(%arg0: !secret.secret<tensor<8xi8>>) {
    ^body(%clear_a: tensor<8xi8>):
      %c = mgmt.modreduce %clear_a : tensor<8xi8>
      %d = mgmt.modreduce %c : tensor<8xi8>
      secret.yield %d : tensor<8xi8>
    // CHECK: } -> (!secret.secret<tensor<8xi8>> {mgmt.mgmt = #mgmt.mgmt<level = 3>})
    } -> !secret.secret<tensor<8xi8>>
    func.return %b : !secret.secret<tensor<8xi8>>
  }
}
