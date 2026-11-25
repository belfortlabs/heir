#ifndef LIB_DIALECT_KERNEL_TRANSFORMS_IMPLEMENTLINEARTRANSFORMS_H_
#define LIB_DIALECT_KERNEL_TRANSFORMS_IMPLEMENTLINEARTRANSFORMS_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {

#define GEN_PASS_DECL_IMPLEMENTLINEARTRANSFORMS
#include "lib/Dialect/Kernel/Transforms/Passes.h.inc"

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_KERNEL_TRANSFORMS_IMPLEMENTLINEARTRANSFORMS_H_
