// Verify Cyclops bootstrap configuration and generated evaluation-key requests.
// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" --mlir-to-ckks="min-slot-count=8192 greedy-level-budget=6 greedy-bootstrap-waterline=3" --scheme-to-cheddar="entry-function=bootstrap runtime=cyclops" %s | FileCheck %s --check-prefix=CYCLOPS
// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" --mlir-to-ckks="min-slot-count=8192 greedy-level-budget=6 greedy-bootstrap-waterline=3" --scheme-to-cheddar="entry-function=bootstrap runtime=cyclops" --cheddar-to-emitc --cheddar-emitc-entry-interface=runtime=cyclops %s | heir-translate --mlir-to-cpp --file-id=client_source | FileCheck %s --check-prefix=CLIENT --implicit-check-not=AddBootstrapRequiredRotations

// Exercise parameter generation and context configuration together. The
// generated Q chain must be deep enough for 4 CtS + 8 EvalMod + 2 StC levels;
// otherwise ConfigureCryptoContext rejects this pipeline before producing the
// setup operations below.

// CYCLOPS: cheddar.create_boot_context
// CYCLOPS-SAME: logMessageRatio = 8
// The key list is planned at compile time and baked in as data. Family 1 is
// conjugation, which only bootstrapping asks for.
// CLIENT: constexpr std::array<KeyRequest, {{[0-9]+}}> kEvaluationKeys
// CLIENT-SAME: {1, 0,
// CLIENT: PrepareRotationKey(GetKeyRequest(

func.func @bootstrap(%input: tensor<1024xf32> {secret.secret})
    -> tensor<1024xf32> {
  %0 = arith.mulf %input, %input : tensor<1024xf32>
  %1 = arith.mulf %0, %0 : tensor<1024xf32>
  %2 = arith.mulf %1, %1 : tensor<1024xf32>
  %3 = arith.mulf %2, %2 : tensor<1024xf32>
  %4 = arith.mulf %3, %3 : tensor<1024xf32>
  %5 = arith.mulf %4, %4 : tensor<1024xf32>
  %result = arith.mulf %5, %input : tensor<1024xf32>
  return %result : tensor<1024xf32>
}
