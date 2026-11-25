#ifndef LIB_TRANSFORMS_LAYOUTOPTIMIZATION_INTERFACEIMPL_H_
#define LIB_TRANSFORMS_LAYOUTOPTIMIZATION_INTERFACEIMPL_H_

#include "lib/Transforms/LayoutOptimization/Hoisting.h"
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project

namespace mlir {
namespace heir {

/// Construct a trivial hoister for which all layouts can be hoisted without
/// any kernel change or difference in the output layout.
Hoister createTrivialHoister(Operation* op);

Hoister createCollapseShapeHoister(tensor::CollapseShapeOp op);
Hoister createPadHoister(tensor::PadOp op);

void registerLayoutConversionHoistableInterface(DialectRegistry& registry);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_LAYOUTOPTIMIZATION_INTERFACEIMPL_H_
