#ifndef LIB_UTILS_LAYOUT_HOISTING_H_
#define LIB_UTILS_LAYOUT_HOISTING_H_

#include <cstdint>

#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {

// Infers a layout relation for the result of an insert_slice operation that
// preserves the slice as a single continuous block in the destination
// ciphertext semantics type.
FailureOr<presburger::IntegerRelation> pushSliceLayoutThroughInsertSlice(
    SmallVector<int64_t> insertSliceSizes, ArrayRef<int64_t> resultShape,
    const presburger::IntegerRelation& sliceLayout);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_HOISTING_H_
