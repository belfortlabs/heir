#ifndef LIB_TRANSFORMS_LAYOUTOPTIMIZATION_INTERFACEIMPL_H_
#define LIB_TRANSFORMS_LAYOUTOPTIMIZATION_INTERFACEIMPL_H_

#include "lib/Transforms/LayoutOptimization/Hoisting.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project

namespace mlir {
namespace heir {

/// Construct a trivial hoister for which all layouts can be hoisted without
/// any kernel change or difference in the output layout.
Hoister createTrivialHoister(Operation* op);

/// Construct a hoister that pre-composes the slot dimension of the Matvec op's
/// matrix with the incremental transformation required to go from vecFromLayout
/// to vecToLayout, while keeping the kernel the same.
Hoister createPrecomposingMatvecHoister(linalg::MatvecOp op);

Hoister createPrecomposingConv1dHoister(linalg::Conv1DOp op);
/// Construct a hoister that hoists a layout conversion through a trivial
/// rank-reducing collapse_shape operation.
Hoister createCollapseShapeHoister(tensor::CollapseShapeOp op);
/// Construct a hoister that hoists a layout conversion through an op that
/// places its source (operand 0) into zeros and moves no data: a zero
/// tensor.pad, or a partial insert into zeros (see isPartialInsertIntoZeros).
/// `sourceToResult` maps each source index to its result index. The other
/// operands need no layout.
Hoister createIntoZerosHoister(Operation* op,
                               presburger::IntegerRelation sourceToResult);

void registerLayoutConversionHoistableInterface(DialectRegistry& registry);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_LAYOUTOPTIMIZATION_INTERFACEIMPL_H_
