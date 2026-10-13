// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" --apply-config-override="config=max_ring_degree=2048" --layout-propagation="min-slot-count=4096" --verify-diagnostics %s

// The 2048 ring holds 1024 slots, fewer than min-slot-count asks for.
// expected-error@+1 {{min-slot-count 4096 exceeds the 1024 slots of the backend's largest ring degree 2048}}
module {
  func.func @add(%arg0: !secret.secret<tensor<8xf32>>) -> !secret.secret<tensor<8xf32>> {
    %0 = secret.generic(%arg0 : !secret.secret<tensor<8xf32>>) {
    ^body(%input0: tensor<8xf32>):
      %1 = arith.addf %input0, %input0 : tensor<8xf32>
      secret.yield %1 : tensor<8xf32>
    } -> !secret.secret<tensor<8xf32>>
    return %0 : !secret.secret<tensor<8xf32>>
  }
}
