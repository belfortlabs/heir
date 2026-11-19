#include "lib/Dialect/Kernel/IR/KernelDialect.h"

#include "lib/Dialect/HEIRInterfaces.h"

// IWYU pragma: begin_keep
#include "lib/Dialect/Kernel/IR/KernelOps.h"
#include "lib/Dialect/Kernel/IR/KernelTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
// IWYU pragma: end_keep

// Generated definitions
#include "lib/Dialect/Kernel/IR/KernelDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Kernel/IR/KernelTypes.cpp.inc"

namespace mlir {
namespace heir {
namespace kernel {

void KernelDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/Kernel/IR/KernelTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Kernel/IR/KernelOps.cpp.inc"
      >();
  // The Cheddar target attaches the level model of kernel.max_pool, which
  // comes from the Cyclops planner (lib/Target/Cheddar/MaxPoolInterfaces.h).
  declarePromisedInterfaces<ReducesLevelOpInterface, MaxPoolOp>();
  declarePromisedInterfaces<BootstrapsInternallyOpInterface, MaxPoolOp>();
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
