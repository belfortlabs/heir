#include <cstdint>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Transforms/LayoutPropagation/Utils.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"         // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using llvm::SmallVector;

TEST(UtilsTest, TestShiftByInserted1) {
  SmallVector<int64_t> dims = {0, 1, 2, 3};
  SmallVector<int64_t> inserts = {1, 2};
  SmallVector<int64_t> expected = {0, 3, 4, 5};
  SmallVector<int64_t> actual = shiftByInserted(dims, inserts);
  EXPECT_EQ(expected, actual);
}

TEST(UtilsTest, TestShiftByInserted2) {
  SmallVector<int64_t> dims = {2, 6, 7, 8};
  SmallVector<int64_t> inserts = {0, 4};
  SmallVector<int64_t> expected = {3, 8, 9, 10};
  SmallVector<int64_t> actual = shiftByInserted(dims, inserts);
  EXPECT_EQ(expected, actual);
}

TEST(UtilsTest, TestShiftByInsertedCollision) {
  SmallVector<int64_t> dims = {3, 6, 7, 8};
  SmallVector<int64_t> inserts = {0, 4};
  SmallVector<int64_t> expected = {5, 8, 9, 10};
  SmallVector<int64_t> actual = shiftByInserted(dims, inserts);
  EXPECT_EQ(expected, actual);
}

TEST(UtilsTest, TestShiftByRemoved1) {
  SmallVector<int64_t> dims = {0, 3, 4, 5};
  SmallVector<int64_t> removals = {1, 2};
  SmallVector<int64_t> expected = {0, 1, 2, 3};
  SmallVector<int64_t> actual = shiftByRemoved(dims, removals);
  EXPECT_EQ(expected, actual);
}

TEST(UtilsTest, TestShiftByRemoved2) {
  SmallVector<int64_t> dims = {3, 8, 9, 10};
  SmallVector<int64_t> removals = {0, 4};
  SmallVector<int64_t> expected = {2, 6, 7, 8};
  SmallVector<int64_t> actual = shiftByRemoved(dims, removals);
  EXPECT_EQ(expected, actual);
}

TEST(UtilsTest, TestShiftByRemovedCollision) {
  SmallVector<int64_t> dims = {5, 8, 9, 10};
  SmallVector<int64_t> removals = {0, 4};
  SmallVector<int64_t> expected = {3, 6, 7, 8};
  SmallVector<int64_t> actual = shiftByRemoved(dims, removals);
  EXPECT_EQ(expected, actual);
}

TEST(UtilsTest, TestReduceLayout) {
  MLIRContext context;
  context.loadDialect<tensor_ext::TensorExtDialect>();

  // Reduce a 4x6 tensor packed into a 3x8 tensor along dimension 0.
  RankedTensorType tensorType =
      RankedTensorType::get({4, 6}, IndexType::get(&context));
  presburger::IntegerRelation relation =
      getRowMajorLayoutRelation(tensorType, 8);
  LayoutAttr layout = LayoutAttr::getFromIntegerRelation(&context, relation);

  SmallVector<int64_t> dimsToReduce = {0};
  LayoutAttr reducedLayout = convertLayoutForReduce(layout, dimsToReduce);
  presburger::IntegerRelation reducedRelation =
      reducedLayout.getIntegerRelation();

  EXPECT_EQ(reducedRelation.getNumDomainVars(), 1);
  EXPECT_EQ(reducedRelation.getNumRangeVars(), 2);

  presburger::IntegerRelation expectedRelation = layout.getIntegerRelation();
  expectedRelation.projectOut(0, 1);
  expectedRelation =
      LayoutAttr::getFromIntegerRelation(&context, expectedRelation)
          .getIntegerRelation();

  EXPECT_TRUE(isRelationEqual(reducedRelation, expectedRelation));
}

TEST(UtilsTest, TestReduceLayoutMultiDim) {
  MLIRContext context;
  context.loadDialect<tensor_ext::TensorExtDialect>();

  // Reduce a 3x2x4 tensor packed into a 3x8 tensor along dimension 2.
  RankedTensorType tensorType =
      RankedTensorType::get({3, 2, 4}, IndexType::get(&context));
  presburger::IntegerRelation relation =
      getRowMajorLayoutRelation(tensorType, 8);
  LayoutAttr layout = LayoutAttr::getFromIntegerRelation(&context, relation);

  SmallVector<int64_t> dimsToReduce = {2};
  LayoutAttr reducedLayout = convertLayoutForReduce(layout, dimsToReduce);
  presburger::IntegerRelation reducedRelation =
      reducedLayout.getIntegerRelation();

  EXPECT_EQ(reducedRelation.getNumDomainVars(), 2);
  EXPECT_EQ(reducedRelation.getNumRangeVars(), 2);

  presburger::IntegerRelation expectedRelation = layout.getIntegerRelation();
  expectedRelation.projectOut(2, 1);
  expectedRelation =
      LayoutAttr::getFromIntegerRelation(&context, expectedRelation)
          .getIntegerRelation();

  EXPECT_TRUE(isRelationEqual(reducedRelation, expectedRelation));
}

TEST(UtilsTest, TestReduceLayoutManyReductions) {
  MLIRContext context;
  context.loadDialect<tensor_ext::TensorExtDialect>();

  // Reduce a 3x2x4 tensor packed into a 3x8 tensor along dimension 1, 2.
  RankedTensorType tensorType =
      RankedTensorType::get({3, 2, 4}, IndexType::get(&context));
  presburger::IntegerRelation relation =
      getRowMajorLayoutRelation(tensorType, 8);
  LayoutAttr layout = LayoutAttr::getFromIntegerRelation(&context, relation);

  SmallVector<int64_t> dimsToReduce = {1, 2};
  LayoutAttr reducedLayout = convertLayoutForReduce(layout, dimsToReduce);
  presburger::IntegerRelation reducedRelation =
      reducedLayout.getIntegerRelation();

  EXPECT_EQ(reducedRelation.getNumDomainVars(), 1);
  EXPECT_EQ(reducedRelation.getNumRangeVars(), 2);

  presburger::IntegerRelation expectedRelation = layout.getIntegerRelation();
  expectedRelation.projectOut(2, 1);
  expectedRelation.projectOut(1, 1);
  expectedRelation =
      LayoutAttr::getFromIntegerRelation(&context, expectedRelation)
          .getIntegerRelation();

  EXPECT_TRUE(isRelationEqual(reducedRelation, expectedRelation));
}

TEST(UtilsTest, TestConvPackingRoundTrip) {
  MLIRContext context;
  context.loadDialect<tensor_ext::TensorExtDialect>();
  OpBuilder builder(&context);
  OwningOpRef<ModuleOp> op = ModuleOp::create(builder.getUnknownLoc());

  auto dataType = RankedTensorType::get({1, 8, 4}, IndexType::get(&context));
  ConvPacking fallback = defaultConv1dPacking(dataType);

  // With nothing recorded, the caller's fallback comes back unchanged.
  ConvPacking absent = getConvPacking(*op, fallback);
  EXPECT_EQ(absent.matrixDataType, dataType);
  EXPECT_EQ(absent.padding, 0);
  EXPECT_EQ(absent.absorbedMatrixWidth, 0);

  ConvPacking recorded = fallback;
  recorded.padding = 2;
  recorded.absorbedMatrixWidth = 1024;
  setConvPacking(*op, recorded);

  ConvPacking readBack = getConvPacking(*op, fallback);
  EXPECT_EQ(readBack.matrixDataType, dataType);
  EXPECT_EQ(readBack.padding, 2);
  EXPECT_TRUE(readBack.interchangeRows);
  EXPECT_EQ(readBack.absorbedMatrixWidth, 1024);

  // A second run replaces the whole packing, so no field can go stale on its
  // own: writing a fresh packing cannot leave the old absorbed width behind.
  setConvPacking(*op, fallback);
  ConvPacking rewritten = getConvPacking(*op, fallback);
  EXPECT_EQ(rewritten.padding, 0);
  EXPECT_EQ(rewritten.absorbedMatrixWidth, 0);
}

}  // namespace
}  // namespace heir
}  // namespace mlir
