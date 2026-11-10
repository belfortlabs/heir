#include "lib/Utils/Layout/InsertIntoZeros.h"

#include <cstdint>

#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {

bool isZeroScalarOrSplat(Value value) {
  return matchPattern(value, m_AnyZeroFloat()) || matchPattern(value, m_Zero());
}

bool isZeroTensor(Value value) {
  if (isZeroScalarOrSplat(value)) return true;
  auto fill = value.getDefiningOp<linalg::FillOp>();
  return fill && isZeroScalarOrSplat(fill.value());
}

bool isSlabInsert(tensor::InsertSliceOp op) {
  for (auto [size, stride, resultSize] :
       llvm::zip_equal(op.getStaticSizes(), op.getStaticStrides(),
                       op.getResultType().getShape())) {
    if (size == 1) continue;
    if (ShapedType::isDynamic(size) || stride != 1 || size != resultSize) {
      return false;
    }
  }
  return true;
}

bool isPartialInsertIntoZeros(tensor::InsertSliceOp op) {
  // Static sizes also make the source shape static, because the verifier
  // makes the source type match them.
  if (!isZeroTensor(op.getDest()) ||
      ShapedType::isDynamicShape(op.getStaticOffsets()) ||
      ShapedType::isDynamicShape(op.getStaticSizes()) ||
      ShapedType::isDynamicShape(op.getStaticStrides()) ||
      !op.getResultType().hasStaticShape()) {
    return false;
  }
  return !isSlabInsert(op);
}

FailureOr<presburger::IntegerRelation> getInsertionRelation(
    tensor::InsertSliceOp op) {
  return getSliceInsertionRelation(op.getSourceType(), op.getResultType(),
                                   SmallVector<int64_t>(op.getStaticOffsets()),
                                   SmallVector<int64_t>(op.getStaticSizes()),
                                   SmallVector<int64_t>(op.getStaticStrides()));
}

}  // namespace heir
}  // namespace mlir
