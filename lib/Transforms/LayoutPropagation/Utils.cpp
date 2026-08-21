#include "lib/Transforms/LayoutPropagation/Utils.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

namespace mlir {
namespace heir {

using ::llvm::ArrayRef;
using ::llvm::SmallVector;

void setConvPacking(Operation* op, const ConvPacking& packing) {
  op->setAttr(tensor_ext::TensorExtDialect::kConvPackingAttrName,
              tensor_ext::ConvPackingAttr::get(
                  op->getContext(), packing.matrixDataType, packing.padding,
                  packing.interchangeRows, packing.absorbedMatrixWidth));
}

ConvPacking getConvPacking(Operation* op, const ConvPacking& fallback) {
  auto attr = op->getAttrOfType<tensor_ext::ConvPackingAttr>(
      tensor_ext::TensorExtDialect::kConvPackingAttrName);
  if (!attr) return fallback;
  return ConvPacking{attr.getMatrixDataType(), attr.getPadding(),
                     attr.getInterchangeRows(), attr.getAbsorbedMatrixWidth()};
}

int64_t maxOfMaxes(ArrayRef<int64_t> d1, ArrayRef<int64_t> d2) {
  int64_t max = d1.front();
  for (int64_t di : d1) {
    max = std::max(max, di);
  }
  for (int64_t di : d2) {
    max = std::max(max, di);
  }
  return max;
}

SmallVector<int64_t> shiftByInserted(ArrayRef<int64_t> dims,
                                     ArrayRef<int64_t> inserts,
                                     bool increment) {
  SmallVector<int64_t> result;
  SmallVector<int64_t> sortedDims(dims);
  SmallVector<int64_t> sortedInserts(inserts);
  llvm::sort(sortedDims);
  llvm::sort(sortedInserts);

  int64_t shift = 0;
  auto dimIt = sortedDims.begin(), insertIt = sortedInserts.begin();
  while (dimIt != sortedDims.end()) {
    auto materializedDim = *dimIt + (increment ? shift : -shift);
    if (insertIt < sortedInserts.end() && *insertIt <= materializedDim) {
      ++insertIt;
      ++shift;
    } else {
      result.push_back(materializedDim);
      ++dimIt;
    }
  }

  return result;
}

SmallVector<int64_t> shiftByRemoved(ArrayRef<int64_t> dims,
                                    ArrayRef<int64_t> removed) {
  return shiftByInserted(dims, removed, false);
}

LayoutAttr convertLayoutForReduce(LayoutAttr inputLayout,
                                  ArrayRef<int64_t> dimsToReduce) {
  std::unique_ptr<presburger::IntegerRelation> clonedRelation =
      inputLayout.getIntegerRelation().clone();

  auto offset = clonedRelation->getVarKindOffset(presburger::VarKind::Domain);
  for (int dim : llvm::reverse(dimsToReduce)) {
    // Project out the reduced dimension.
    auto dimIndex = offset + dim;
    assert(clonedRelation->getVarKindAt(dimIndex) ==
           presburger::VarKind::Domain);
    clonedRelation->projectOut(dimIndex, 1);
  }

  MLIRContext* context = inputLayout.getContext();
  return LayoutAttr::getFromIntegerRelation(context,
                                            std::move(*clonedRelation));
}

}  // namespace heir
}  // namespace mlir
