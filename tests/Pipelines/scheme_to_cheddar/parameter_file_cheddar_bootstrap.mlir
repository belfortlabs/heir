// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" --mlir-to-ckks="min-slot-count=8192 greedy-level-budget=17 greedy-bootstrap-waterline=1 cheddar-parameter-file=%S/../../Parameters/cheddar/bootparam_35.json" --scheme-to-cheddar="entry-function=bootstrap" %s | FileCheck %s
// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" --mlir-to-ckks="min-slot-count=8192 greedy-level-budget=17 greedy-bootstrap-waterline=1 cheddar-parameter-file=%S/../../Parameters/cheddar/bootparam_35.json" --scheme-to-cheddar="entry-function=bootstrap" --cheddar-to-emitc --cheddar-emitc-entry-interface %s | heir-translate --mlir-to-cpp --file-id=source | FileCheck %s --check-prefix=SOURCE

// A bootstrapping program on CHEDDAR's 32-bit chain and the scale-snu
// runtime: the level budget of 17 spans levels 0..16, where the bootstrap
// lands (3 StC levels below the file's default encryption level 19). The
// runtime Parameter keeps the file's default level, as scale-snu's
// BootContext requires, and the message ratio is derived from the file's
// q0 headroom (floor(15.0 - 2) = 13).

// CHECK: cheddar.make_parameter {parameterSet = #cheddar.parameter_set<logN = 16, logScale = 35
// CHECK-SAME: wordBits = 32
// CHECK-SAME: defaultEncryptionLevel = 19
// CHECK-SAME: sparseHammingWeight = 32>
// CHECK: cheddar.create_boot_context
// CHECK-SAME: config = #cheddar.bootstrap_config<numCtsLevels = 4, numStcLevels = 3, numEvalModLevels = 8, logMessageRatio = 13>

// SOURCE: using word = std::uint32_t;
// SOURCE: BootContext<word>::Create(cheddar_param, BootParameter(cheddar_param.max_level_, 4, 3, 13));

func.func @bootstrap(%input: tensor<1024xf32> {secret.secret})
    -> tensor<1024xf32> {
  %0 = arith.mulf %input, %input : tensor<1024xf32>
  %1 = arith.mulf %0, %0 : tensor<1024xf32>
  %2 = arith.mulf %1, %1 : tensor<1024xf32>
  %3 = arith.mulf %2, %2 : tensor<1024xf32>
  %4 = arith.mulf %3, %3 : tensor<1024xf32>
  %5 = arith.mulf %4, %4 : tensor<1024xf32>
  %6 = arith.mulf %5, %5 : tensor<1024xf32>
  %7 = arith.mulf %6, %6 : tensor<1024xf32>
  %8 = arith.mulf %7, %7 : tensor<1024xf32>
  %9 = arith.mulf %8, %8 : tensor<1024xf32>
  %10 = arith.mulf %9, %9 : tensor<1024xf32>
  %11 = arith.mulf %10, %10 : tensor<1024xf32>
  %12 = arith.mulf %11, %11 : tensor<1024xf32>
  %13 = arith.mulf %12, %12 : tensor<1024xf32>
  %14 = arith.mulf %13, %13 : tensor<1024xf32>
  %15 = arith.mulf %14, %14 : tensor<1024xf32>
  %result = arith.mulf %15, %input : tensor<1024xf32>
  return %result : tensor<1024xf32>
}
