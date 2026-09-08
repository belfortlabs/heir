#ifndef LIB_DIALECT_CHEDDAR_IR_CHEDDARRUNTIME_H_
#define LIB_DIALECT_CHEDDAR_IR_CHEDDARRUNTIME_H_

#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

namespace mlir::heir::cheddar {

constexpr StringLiteral kCheddarRuntimeAttrName = "cheddar.runtime";
constexpr StringLiteral kCyclopsRuntimeName = "cyclops";
constexpr StringLiteral kScaleSnuRuntimeName = "cheddar";

// Returns the runtime name recorded on the module, or a null attribute when no
// CHEDDAR pass has tagged the module yet.
inline StringAttr getCheddarRuntime(Operation* op) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module) return StringAttr{};
  return module->getAttrOfType<StringAttr>(kCheddarRuntimeAttrName);
}

inline bool useCyclopsRuntime(Operation* op) {
  auto runtime = getCheddarRuntime(op);
  return runtime && runtime.getValue() == kCyclopsRuntimeName;
}

}  // namespace mlir::heir::cheddar

#endif  // LIB_DIALECT_CHEDDAR_IR_CHEDDARRUNTIME_H_
