// RUN: heir-opt %s --annotate-module="backend=cheddar scheme=ckks" --mlir-to-ckks="min-slot-count=4096" --scheme-to-cheddar="entry-function=split runtime=cyclops" --cheddar-to-emitc --cheddar-emitc-entry-interface="runtime=cyclops" | heir-translate --mlir-to-cpp --file-id=server_source | FileCheck %s --implicit-check-not=UserInterface --implicit-check-not=SecretKey

// CHECK: const heir::cyclops::DebugSink*
// CHECK: heir::cyclops::emitCheckpoint
// CHECK: split/input/0
// CHECK: heir::cyclops::emitCheckpoint
// CHECK: split/output/0
// CHECK: EncryptedOutputs Evaluate
// CHECK: const DebugSink*

func.func @split(%input: tensor<4xf32> {secret.secret}) -> tensor<4xf32> {
  debug.validate %input {name = "split/input/0"} : tensor<4xf32>
  %result = arith.mulf %input, %input : tensor<4xf32>
  debug.validate %result {name = "split/output/0"} : tensor<4xf32>
  return %result : tensor<4xf32>
}
