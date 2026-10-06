// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" --mlir-to-ckks="min-slot-count=8192 greedy-level-budget=15 greedy-bootstrap-waterline=1 cheddar-parameter-file=%S/../../Parameters/cyclops/bootparam_35bit_14lvl.json" --scheme-to-cheddar="entry-function=bootstrap runtime=cyclops" %s | FileCheck %s
// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" --mlir-to-ckks="min-slot-count=8192 greedy-level-budget=15 greedy-bootstrap-waterline=1 cheddar-parameter-file=%S/../../Parameters/cyclops/bootparam_35bit_14lvl.json" --scheme-to-cheddar="entry-function=bootstrap runtime=cyclops" --cheddar-to-emitc --cheddar-emitc-entry-interface=runtime=cyclops %s | heir-translate --mlir-to-cpp --file-id=server_source | FileCheck %s --check-prefix=SERVER

// A bootstrapping program on the 32-bit chain: the level budget of 15 spans
// the file's residual levels 0..14, so the bootstrap lands where HEIR expects
// it (level 14, the file's default encryption level). The
// bootstrap split (5 CtS, 8 EvalMod, 3 StC) and the EvalMod approximation come
// from the file, and the key planner accepts them (it checks that the
// approximation consumes the eight levels the file reserves). Family 1 in the
// planned keys is the conjugation key only a bootstrap asks for. The file
// states no message ratio, so it is derived from its 49.8-bit q0 over the
// 35-bit scale: floor(14.8 - 2) = 12.

// CHECK: func.func @bootstrap__server_setup
// CHECK: cheddar.make_parameter {parameterSet = #cheddar.parameter_set<logN = 16, logScale = 35
// CHECK-SAME: wordBits = 32
// CHECK-SAME: defaultEncryptionLevel = 14
// CHECK-SAME: sparseHammingWeight = 32>
// CHECK: cheddar.create_boot_context
// CHECK-SAME: config = #cheddar.bootstrap_config<numCtsLevels = 5, numStcLevels = 3, numEvalModLevels = 8, logMessageRatio = 12, evalMod = <type = "cos_hk_even", degree = 26, interval = 16, logIntervalReduction = 3, invDegree = 0>>
// CHECK: func.func @bootstrap__setup
// CHECK-SAME: cheddar.evaluation_keys = array<i64:
// CHECK-SAME: 1, 0,

// SERVER: using word = std::uint32_t;
// SERVER: Mod1ParametersLiteral _boot_mod1 = BootParameter::DefaultMod1(12);
// SERVER: _boot_mod1.type = Mod1Type::kCosHKEven;
// SERVER: _boot_mod1.degree = 26;
// SERVER: _boot_mod1.interval = 16;
// SERVER: _boot_mod1.log_interval_reduction = 3;
// SERVER: _boot_mod1.inv_degree = 0;
// SERVER: BootContext<word>::Create(cheddar_param, BootParameter(cheddar_param.max_level_, 5, 3, 12, _boot_mod1));

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
