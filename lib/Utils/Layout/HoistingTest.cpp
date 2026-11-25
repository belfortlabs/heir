#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Layout/Codegen.h"
#include "lib/Utils/Layout/Hoisting.h"
#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"    // from @llvm-project

// ISL

namespace mlir {
namespace heir {
namespace {

using presburger::IntegerRelation;

TEST(HoistingTest, PushThroughInsertSlice) {
  // Insert a 4x4 slice (from a 32 slot ciphertext) into a 1x2x4x4 tensor. The
  // result tensor should have two ciphertexts.
  MLIRContext context;

  auto maybeSliceLayout = getIntegerRelationFromIslStr(
      "{ [i0, i1] -> [ct, slot] : ct = 0 and (-4i0 - i1 + slot) mod 16 = 0 and "
      "0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= slot <= 31 }");
  ASSERT_TRUE(succeeded(maybeSliceLayout));

  auto actual = pushSliceLayoutThroughInsertSlice({1, 1, 4, 4}, {1, 2, 4, 4},
                                                  maybeSliceLayout.value());
  ASSERT_TRUE(succeeded(actual));

  // The result layout is a 4 dimension to ct, slot. Enumerate the points. Some
  // domain points can be mapped to many range points.
  PointPairCollector collector(4, 2);
  enumeratePoints(actual.value(), collector);
  std::map<std::vector<int64_t>, std::vector<std::vector<int64_t>>> pointMap;
  for (const auto& [domain, range] : collector.points) {
    if (!pointMap.contains(domain)) {
      pointMap[domain] = std::vector<std::vector<int64_t>>({range});
    }
    pointMap[domain].push_back(range);
  }

  std::vector<std::vector<int64_t>> expectedDomain = {
      {0, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 2, 1},
      {0, 1, 0, 0}, {0, 1, 1, 0}, {0, 1, 2, 1},
  };
  std::vector<std::vector<int64_t>> expectedRange = {
      {0, 0}, {0, 4}, {0, 9}, {1, 0}, {1, 4}, {1, 9},
  };

  for (const auto& [domain, range] : llvm::zip(expectedDomain, expectedRange)) {
    if (!pointMap.contains(domain)) {
      FAIL() << "Failed to find point (" << ::testing::PrintToString(domain)
             << ", " << ::testing::PrintToString(range)
             << ") in actual relation.";
    }
    if (!llvm::is_contained(pointMap[domain], range)) {
      FAIL() << "Expected point at " << ::testing::PrintToString(domain)
             << " to be " << ::testing::PrintToString(range) << ", got "
             << ::testing::PrintToString(pointMap[domain]);
    }
  }

  // There should be two ciphertexts in the result, each corresponding to two
  // slices in the second dimension (1x2x4x4).
  auto ctBound = actual.value().getConstantBound64(
      presburger::BoundType::UB,
      actual.value().getVarKindOffset(presburger::VarKind::Range));
  ASSERT_TRUE(ctBound.has_value());
  EXPECT_EQ(ctBound.value(), 1);  // inclusive
}

TEST(HoistingTest, PushThroughInsertSlice1x8Into2x8) {
  // Insert a 1x8 slice into a 2x8 tensor. This test reproduces a bug where
  // an extra domain variable was added.
  MLIRContext context;

  // Simple layout: 1x8 tensor where each element maps to a slot in ciphertext 0
  auto maybeSliceLayout = getIntegerRelationFromIslStr(
      "{ [i0, i1] -> [ct, slot] : ct = 0 and slot = i1 and i0 = 0 and "
      "0 <= i1 <= 7 }");
  ASSERT_TRUE(succeeded(maybeSliceLayout));

  auto actual = pushSliceLayoutThroughInsertSlice({1, 8}, {2, 8},
                                                  maybeSliceLayout.value());
  ASSERT_TRUE(succeeded(actual));

  // The result should have 2 domain dimensions (matching the 2x8 destination)
  // not 3 dimensions.
  EXPECT_EQ(actual.value().getNumDomainVars(), 2);
  EXPECT_EQ(actual.value().getNumRangeVars(), 2);
}

TEST(HoistingTest, PushThroughInsertSliceCtUpperBound) {
  // Test that the ct (ciphertext) variable has the correct upper bound when
  // inserting a slice. This test verifies the fix for missing upper bound on
  // the ct variable.
  MLIRContext context;

  // Insert a 1x4 slice into a 3x4 tensor. The first dimension is a unit
  // dimension, so we expect ct to range from 0 to 2 (product of unit dims = 3).
  auto maybeSliceLayout = getIntegerRelationFromIslStr(
      "{ [i0, i1] -> [ct, slot] : ct = 0 and slot = i1 and i0 = 0 and "
      "0 <= i1 <= 3 }");
  ASSERT_TRUE(succeeded(maybeSliceLayout));

  auto actual = pushSliceLayoutThroughInsertSlice({1, 4}, {3, 4},
                                                  maybeSliceLayout.value());
  ASSERT_TRUE(succeeded(actual));

  // Check that the ct variable has upper bound of 2 (3 slices - 1).
  auto ctBound = actual.value().getConstantBound64(
      presburger::BoundType::UB,
      actual.value().getVarKindOffset(presburger::VarKind::Range));
  ASSERT_TRUE(ctBound.has_value());
  EXPECT_EQ(ctBound.value(), 2);  // inclusive upper bound
}

TEST(HoistingTest, PushThroughInsertSliceMultipleUnitDimsCtBound) {
  // Test ct upper bound with multiple unit dimensions.
  // Insert a 1x1x8 slice into a 2x3x8 tensor. The first two dimensions are
  // unit dimensions, so ct should range from 0 to 5 (product = 2*3 = 6).
  MLIRContext context;

  auto maybeSliceLayout = getIntegerRelationFromIslStr(
      "{ [i0, i1, i2] -> [ct, slot] : ct = 0 and slot = i2 and i0 = 0 and "
      "i1 = 0 and 0 <= i2 <= 7 }");
  ASSERT_TRUE(succeeded(maybeSliceLayout));

  auto actual = pushSliceLayoutThroughInsertSlice({1, 1, 8}, {2, 3, 8},
                                                  maybeSliceLayout.value());
  ASSERT_TRUE(succeeded(actual));

  // Check that the ct variable has upper bound of 5 (2*3 - 1).
  auto ctBound = actual.value().getConstantBound64(
      presburger::BoundType::UB,
      actual.value().getVarKindOffset(presburger::VarKind::Range));
  ASSERT_TRUE(ctBound.has_value());
  EXPECT_EQ(ctBound.value(), 5);  // inclusive upper bound
}

}  // namespace
}  // namespace heir
}  // namespace mlir
