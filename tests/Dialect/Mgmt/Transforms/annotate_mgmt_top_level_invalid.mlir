// RUN: heir-opt --annotate-module="backend=lattigo" --annotate-mgmt --verify-diagnostics %s

// A pinned top below the program's depth cannot hold the program.
// expected-error@below {{the program is 2 levels deep, but mgmt.top_level pins its top at level 1}}
module attributes {mgmt.top_level = 1 : i64} {
  func.func @main(%arg0: !secret.secret<tensor<8xi8>>) -> !secret.secret<tensor<8xi8>> {
    %b = secret.generic(%arg0: !secret.secret<tensor<8xi8>>) {
    ^body(%clear_a: tensor<8xi8>):
      %c = mgmt.modreduce %clear_a : tensor<8xi8>
      %d = mgmt.modreduce %c : tensor<8xi8>
      secret.yield %d : tensor<8xi8>
    } -> !secret.secret<tensor<8xi8>>
    func.return %b : !secret.secret<tensor<8xi8>>
  }
}
