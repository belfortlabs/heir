// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" --mlir-to-ckks="min-slot-count=1024 cheddar-parameter-file=%S/../../Parameters/cyclops/bootparam_35bit_14lvl.json" --scheme-to-cheddar="entry-function=main runtime=cyclops" %s | FileCheck %s
// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" --mlir-to-ckks="min-slot-count=1024 cheddar-parameter-file=%S/../../Parameters/cyclops/bootparam_35bit_14lvl.json" --scheme-to-cheddar="entry-function=main runtime=cyclops" --cheddar-to-emitc --cheddar-emitc-entry-interface=runtime=cyclops %s | heir-translate --mlir-to-cpp --file-id=client_source | FileCheck %s --check-prefix=CLIENT

// A 32-bit word chain from a Cyclops parameter file, end to end: the two
// multiplications need three levels, which the logN 13 ring of the file
// provides; the client is generated for 32-bit words with the file's own
// primes, level layout and key-switching policy, and the evaluation keys are
// planned against that very parameter set: the two multiplications
// relinearize at levels 2 and 1, so the client requests a multiplication key
// for each (default-preferred, family 2, mode 1), which a ring without a
// default key turns into level-specific keys.

// CHECK: module attributes
// CHECK-SAME: cheddar.word_bits = 32 : i64
// CHECK: func.func @main__setup
// CHECK-SAME: cheddar.evaluation_keys = array<i64: 2, 0, {{[12]}}, 1, -1, 2, 0, {{[12]}}, 1, -1>
// CHECK: cheddar.make_parameter {parameterSet = #cheddar.parameter_set<logN = 13, logScale = 35
// CHECK-SAME: terminalPrimes = [30539777, 32899073, 29884417, 31326209, 36175873]
// CHECK-SAME: levelConfig = [0, 2, 2, 1, 4, 0]
// CHECK-SAME: wordBits = 32
// CHECK-SAME: defaultEncryptionLevel = 2

// CLIENT: using word = std::uint32_t;
// CLIENT: static Parameter<word> cheddar_param(13, static_cast<double>(UINT64_C(1) << 35), 2, std::vector<std::pair<int, int>>{{[{][{]}}0, 2}, {2, 1}, {4, 0}}, std::vector<word>{969146369ULL,
// CLIENT-SAME: std::vector<word>{30539777ULL, 32899073ULL, 29884417ULL, 31326209ULL, 36175873ULL}, std::pair<int, int>{0, 0}, 12);
// CLIENT: cheddar_param.SetDenseHammingWeight(128);
// CLIENT: cheddar_param.SetSparseHammingWeight(128);
// CLIENT: cheddar_param.SetMaxLogPQ(216.25);
// CLIENT: cheddar_param.SetLevelSpecificKS(true);
// CLIENT: cheddar_param.SetMaxKeySwitchAux(23);

func.func @main(%input: tensor<1024xf32> {secret.secret}) -> tensor<1024xf32> {
  %0 = arith.mulf %input, %input : tensor<1024xf32>
  %1 = arith.mulf %0, %0 : tensor<1024xf32>
  return %1 : tensor<1024xf32>
}
