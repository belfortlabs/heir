// RUN: heir-opt '--cheddar-configure-crypto-context=entry-function=main use-cyclops-runtime=true' %s | FileCheck %s
// RUN: heir-opt '--cheddar-configure-crypto-context=entry-function=main use-cyclops-runtime=true log-message-ratio=9' %s | FileCheck %s --check-prefix=RATIO
// RUN: heir-opt '--cheddar-configure-crypto-context=entry-function=main use-cyclops-runtime=true' --cheddar-bufferize --fold-memref-alias-ops --canonicalize --convert-to-emitc=filter-dialects=cheddar,arith,scf --cheddar-emitc-boundary --reconcile-unrealized-casts %s | FileCheck %s --check-prefix=EMITC

// A module whose modulus chain was imported from a Cyclops parameter file
// (generate-param-ckks with cheddar-parameter-file): the parameter set and the
// bootstrap split are taken as recorded, and the word width is left for the
// EmitC entry interface. The chain below is a cut-down 32-bit layout: two
// residual levels and a one-level bootstrap circuit above them.

!boot_context = !cheddar.boot_context
!ciphertext = !cheddar.ciphertext
!evk_map = !cheddar.evk_map
!ui = !cheddar.user_interface

module attributes {
  cheddar.bootstrap_config = #cheddar.bootstrap_config<numCtsLevels = 1, numStcLevels = 0, numEvalModLevels = 0, evalMod = #cheddar.eval_mod<type = "cos_hk_even", degree = 26>>,
  cheddar.parameter_set = #cheddar.parameter_set<logN = 16, logScale = 35, mainPrimes = [969146369, 1172439041, 958136321], auxPrimes = [964558849, 974258177], terminalPrimes = [30539777, 32899073], levelConfig = [0, 2, 2, 1, 3, 1], wordBits = 32, defaultEncryptionLevel = 1, defaultNumAux = 1, levelSpecificKs = true, maxKeySwitchAux = 2, maxLogPq = 1730.0 : f64, denseHammingWeight = 1024, sparseHammingWeight = 32>,
  ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [1125899908022273, 35184372121601, 1152921504606994433], P = [1073741827], logDefaultScale = 35>,
  scheme.actual_slot_count = 32768 : i64,
  scheme.requested_slot_count = 8 : i64
} {
  func.func @main(%ctx: !boot_context, %ui: !ui, %ct: tensor<!ciphertext>, %evk: !evk_map) -> tensor<!ciphertext> {
    %rotDest = bufferization.alloc_tensor() : tensor<!ciphertext>
    %rotated = cheddar.hrot %ctx, %evk, %ct, %rotDest {level = 1 : i64, static_distance = 7 : i64} : (!boot_context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dest = bufferization.alloc_tensor() : tensor<!ciphertext>
    %result = cheddar.boot %ctx, %rotated, %evk, %dest : (!boot_context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
    return %result : tensor<!ciphertext>
  }
}

// CHECK: module attributes
// CHECK-SAME: cheddar.logDefaultScale = 35 : i64
// CHECK-SAME: cheddar.logN = 16 : i64
// CHECK-SAME: cheddar.word_bits = 32 : i64
// CHECK-NOT: cheddar.parameter_set
// CHECK: func.func @main__server_setup
// CHECK: cheddar.make_parameter {parameterSet = #cheddar.parameter_set<logN = 16, logScale = 35, mainPrimes = [969146369, 1172439041, 958136321], auxPrimes = [964558849, 974258177], terminalPrimes = [30539777, 32899073], levelConfig = [0, 2, 2, 1, 3, 1], wordBits = 32, defaultEncryptionLevel = 1, defaultNumAux = 1, levelSpecificKs = true, maxKeySwitchAux = 2, maxLogPq = 1.730000e+03 : f64, denseHammingWeight = 1024, sparseHammingWeight = 32>
// CHECK: cheddar.create_boot_context
// CHECK-SAME: config = #cheddar.bootstrap_config<numCtsLevels = 1, numStcLevels = 0, numEvalModLevels = 0, evalMod = <type = "cos_hk_even", degree = 26>>
// CHECK: cheddar.prepare_bootstrap_context
// CHECK-SAME: numSlots = 256
// CHECK: func.func @main__setup
// CHECK-SAME: cheddar.bootstrap_config = #cheddar.bootstrap_config<numCtsLevels = 1, numStcLevels = 0, numEvalModLevels = 0, evalMod = <type = "cos_hk_even", degree = 26>>
// CHECK-SAME: cheddar.bootstrap_slots = 256 : i64
// CHECK-SAME: cheddar.rotation_keys = array<i64: 7, 1>

// An explicit log-message-ratio option overrides the file's headroom.
// RATIO: cheddar.create_boot_context
// RATIO-SAME: logMessageRatio = 9

// EMITC: emitc.verbatim "static Parameter<word> cheddar_param(16, static_cast<double>(UINT64_C(1) << 35), 1, std::vector<std::pair<int, int>>{{[{][{]}}0, 2}, {2, 1}, {3, 1}}, std::vector<word>{969146369ULL, 1172439041ULL, 958136321ULL}, std::vector<word>{964558849ULL, 974258177ULL}, std::vector<word>{30539777ULL, 32899073ULL}, std::pair<int, int>{0, 0}, 1);"
// EMITC: emitc.verbatim "cheddar_param.SetDenseHammingWeight(1024);"
// EMITC: emitc.verbatim "cheddar_param.SetSparseHammingWeight(32);"
// EMITC: emitc.verbatim "cheddar_param.SetMaxLogPQ(1730);"
// EMITC: emitc.verbatim "cheddar_param.SetLevelSpecificKS(true);"
// EMITC: emitc.verbatim "cheddar_param.SetMaxKeySwitchAux(2);"
// EMITC: emitc.verbatim "{"
// EMITC: emitc.verbatim "Mod1ParametersLiteral _boot_mod1 = BootParameter::DefaultMod1(BootParameter::kDefaultLogMessageRatio);"
// EMITC: emitc.verbatim "_boot_mod1.type = Mod1Type::kCosHKEven;"
// EMITC: emitc.verbatim "_boot_mod1.degree = 26;"
// EMITC: emitc.verbatim "{} = BootContext<word>::Create({}, BootParameter({}.max_level_, 1, 0, BootParameter::kDefaultLogMessageRatio, _boot_mod1));"
// EMITC: emitc.verbatim "}"
