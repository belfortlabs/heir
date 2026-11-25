#ifndef LIB_TRANSFORMS_LINEARKERNEL_STRIDEDLAYOUTPROPAGATION_H_
#define LIB_TRANSFORMS_LINEARKERNEL_STRIDEDLAYOUTPROPAGATION_H_

#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/LinearKernel/StridedLayoutPropagation.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/LinearKernel/StridedLayoutPropagation.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_LINEARKERNEL_STRIDEDLAYOUTPROPAGATION_H_
