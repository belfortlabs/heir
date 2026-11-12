#ifndef LIB_UTILS_LAYOUT_INSERTINTOZEROS_H_
#define LIB_UTILS_LAYOUT_INSERTINTOZEROS_H_

#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {

// Whether `value` is a zero scalar or a splat zero constant, integer or float.
bool isZeroScalarOrSplat(Value value);

// Whether `value` is a tensor of zeros: a splat zero constant, or a
// `linalg.fill` of a zero scalar.
bool isZeroTensor(Value value);

// Whether the slice of `op` fills whole slabs of the result: every slice
// dimension is 1, or it is the full result dimension with stride 1. The
// general insert_slice path in layout propagation supports only such an
// insert.
bool isSlabInsert(tensor::InsertSliceOp op);

// Whether `op` inserts its source into a tensor of zeros, at static offsets,
// sizes and strides, and the slice does not fill whole slabs of the result
// (see isSlabInsert). Like a zero `tensor.pad`, such an insert moves no data:
// the result layout is the source layout seen through the insertion, and the
// zero positions stay unmapped. torch-mlir lowers a ConvTranspose this way: it
// spreads the input over a zero tensor with the stride, then runs a stride-1
// convolution.
//
// A slab insert into zeros keeps the general insert_slice path, which gives
// the result the full layout of the packed zeros. A concat lowers to a chain
// of slab inserts, and each later insert in the chain needs that full layout
// for its own slab.
bool isPartialInsertIntoZeros(tensor::InsertSliceOp op);

// The relation source index -> result index of an insert with static offsets,
// sizes and strides.
FailureOr<presburger::IntegerRelation> getInsertionRelation(
    tensor::InsertSliceOp op);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_INSERTINTOZEROS_H_
