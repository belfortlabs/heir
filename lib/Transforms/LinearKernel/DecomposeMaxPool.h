#ifndef LIB_TRANSFORMS_LINEARKERNEL_DECOMPOSEMAXPOOL_H_
#define LIB_TRANSFORMS_LINEARKERNEL_DECOMPOSEMAXPOOL_H_

#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/LinearKernel/DecomposeMaxPool.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/LinearKernel/DecomposeMaxPool.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_LINEARKERNEL_DECOMPOSEMAXPOOL_H_
