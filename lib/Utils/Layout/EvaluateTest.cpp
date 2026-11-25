#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Layout/Evaluate.h"
#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using presburger::IntegerRelation;
using ::testing::Eq;

TEST(EvaluateTest, EvaluateLayoutOnVectorTrivial) {
  IntegerRelation relation =
      getIntegerRelationFromIslStr(
          "{ [d0] -> [ct, d1] : d0 - d1 = 0 and d0 >= 0 and d1 >= 0 and 10 >= "
          "d0 and 10 >= d1 and ct = 0 }")
          .value();
  std::vector<int> input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto result = evaluateLayoutOnVector(relation, input);
  ASSERT_THAT(result.size(), Eq(1));
  ASSERT_THAT(result[0], Eq(input));
}

TEST(EvaluateTest, EvaluateLayoutOnVectorRowMajor) {
  // Data of size 32 being packed into ciphertexts of size 1024.
  auto relation =
      getIntegerRelationFromIslStr(
          "{ [row] -> [ct, slot] : (slot - row) mod 32 = 0 and row >= 0 and ct "
          ">= 0 and slot >= 0 and 1023 >= slot and 0 >= ct and 31 >= row }")
          .value();

  std::vector<int> input;
  input.reserve(32);
  for (int i = 0; i < 32; i++) {
    input.push_back(i);
  }

  auto result = evaluateLayoutOnVector(relation, input);
  ASSERT_THAT(result.size(), Eq(1));
  ASSERT_THAT(result[0].size(), Eq(1024));
  for (int i = 0; i < 32; i++) {
    std::vector<int> slice = {result[0].begin() + i * 32,
                              result[0].begin() + i * 32 + 32};
    ASSERT_THAT(slice, Eq(input));
  }
}

}  // namespace
}  // namespace heir
}  // namespace mlir
