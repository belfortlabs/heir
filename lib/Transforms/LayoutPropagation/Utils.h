#ifndef LIB_TRANSFORMS_LAYOUTPROPAGATION_UTILS_H_
#define LIB_TRANSFORMS_LAYOUTPROPAGATION_UTILS_H_

#include <cstdint>
#include <optional>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Utils/Layout/Convolution.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"       // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

namespace mlir {
namespace heir {

// Records `packing` on `op`, replacing whatever packing was there. Every field
// travels in one attribute, so a run cannot leave half of a stale packing
// behind for the next pass to act on.
void setConvPacking(Operation* op, const ConvPacking& packing);

// The packing recorded on `op`, or `fallback` when `op` carries none.
//
// LayoutPropagation always records one, so a missing packing means the op never
// went through it: hand written test IR, for instance. `fallback` is then the
// packing a convolution has before any fold or absorption, which is what such
// IR describes.
ConvPacking getConvPacking(Operation* op, const ConvPacking& fallback);

using tensor_ext::LayoutAttr;

int64_t maxOfMaxes(::llvm::ArrayRef<int64_t> d1, ::llvm::ArrayRef<int64_t> d2);

// `dims` is a list of dims of a tensor, and `inserts` represents a list of
// inserted dims in that tensor (via alignment attr's insertedDims). This
// function shifts the values of `dims` as if the dims from `inserts` were
// inserted.
//
// Example:
//
// input = [0, 1, 2, 3]
// inserts = [1, 2]
//
// output = [0, 3, 4, 5]
//
// This allows you to track how `dims` maps to new dims as a result of the
// inserts.
::llvm::SmallVector<int64_t> shiftByInserted(::llvm::ArrayRef<int64_t> dims,
                                             ::llvm::ArrayRef<int64_t> inserts,
                                             bool increment = true);

// Map the dims in the `dims` list to new dims when the dims in `removed` are
// removed from the tensor.
//
// Assumes dims and removed do not have any common values
//
// Example:
// input = [0, 3, 4, 5]
// removed = [1, 2]
//
// output = [0, 1, 2, 3]
::llvm::SmallVector<int64_t> shiftByRemoved(::llvm::ArrayRef<int64_t> dims,
                                            ::llvm::ArrayRef<int64_t> removed);

// A helper to convert the layout of an input tensor to a reduce op. The result
// layout is equivalent to setting the summed dimensions to 0.
LayoutAttr convertLayoutForReduce(LayoutAttr inputLayout,
                                  ArrayRef<int64_t> dimsToReduce);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_LAYOUTPROPAGATION_UTILS_H_
