// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" --mlir-to-ckks="min-slot-count=1024 cheddar-parameter-file=%S/../../Parameters/cheddar/bootparam_35.json" --scheme-to-cheddar="entry-function=main" %s | FileCheck %s
// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" --mlir-to-ckks="min-slot-count=1024 cheddar-parameter-file=%S/../../Parameters/cheddar/bootparam_35.json" --scheme-to-cheddar="entry-function=main" --cheddar-to-emitc --cheddar-emitc-entry-interface %s | heir-translate --mlir-to-cpp --file-id=source | FileCheck %s --check-prefix=SOURCE

// A 32-bit word chain from CHEDDAR's own bootstrapping parameters, end to end
// on the scale-snu runtime: the two multiplications need three levels, and
// the generated code builds CHEDDAR's Parameter for 32-bit words from the
// file's primes and its whole level layout, encrypting at level 2.

// CHECK: module attributes
// CHECK-SAME: cheddar.word_bits = 32 : i64
// CHECK: cheddar.make_parameter {parameterSet = #cheddar.parameter_set<logN = 16, logScale = 35
// CHECK-SAME: terminalPrimes = [30539777, 37224449, 33292289, 32899073, 35389441]
// CHECK-SAME: levelConfig = [0, 2, 2, 1, 4, 0, 1, 5, 3, 4, 5, 3, 7, 2, 9, 1, 11, 0, 8, 5, 10, 4, 12, 3, 14, 2, 16, 1, 18, 0, 15, 5, 17, 4, 19, 3, 21, 2, 23, 1, 25, 1, 27, 1, 29, 1, 31, 1, 33, 1, 35, 1, 37, 1, 39, 1, 41, 1, 43, 1, 43, 3, 43, 5]
// CHECK-SAME: wordBits = 32
// CHECK-SAME: defaultEncryptionLevel = 2

// SOURCE: using word = std::uint32_t;
// SOURCE: static Parameter<word> cheddar_param(16, static_cast<double>(UINT64_C(1) << 35), 2, std::vector<std::pair<int, int>>{{[{][{]}}0, 2}, {2, 1}, {4, 0}, {1, 5}, {3, 4}, {5, 3}, {7, 2}, {9, 1}, {11, 0}, {8, 5}, {10, 4}, {12, 3}, {14, 2}, {16, 1}, {18, 0}, {15, 5}, {17, 4}, {19, 3}, {21, 2}, {23, 1}, {25, 1}, {27, 1}, {29, 1}, {31, 1}, {33, 1}, {35, 1}, {37, 1}, {39, 1}, {41, 1}, {43, 1}, {43, 3}, {43, 5}}, std::vector<word>{1060765697ULL,
// SOURCE-SAME: std::vector<word>{30539777ULL, 37224449ULL, 33292289ULL, 32899073ULL, 35389441ULL});
// SOURCE: cheddar_param.SetDenseHammingWeight(32768);
// SOURCE: cheddar_param.SetSparseHammingWeight(32768);
// SOURCE-NOT: SetMaxLogPQ

func.func @main(%input: tensor<1024xf32> {secret.secret}) -> tensor<1024xf32> {
  %0 = arith.mulf %input, %input : tensor<1024xf32>
  %1 = arith.mulf %0, %0 : tensor<1024xf32>
  return %1 : tensor<1024xf32>
}
