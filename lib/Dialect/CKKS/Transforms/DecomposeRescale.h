#ifndef LIB_DIALECT_CKKS_TRANSFORMS_DECOMPOSE_RESCALE_H_
#define LIB_DIALECT_CKKS_TRANSFORMS_DECOMPOSE_RESCALE_H_

#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace ckks {

#define GEN_PASS_DECL_DECOMPOSERESCALE
#include "lib/Dialect/CKKS/Transforms/Passes.h.inc"

struct DecomposeRescalePattern : public OpRewritePattern<RescaleOp> {
  using OpRewritePattern<RescaleOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(RescaleOp op,
                                PatternRewriter& rewriter) const override;
};

}  // namespace ckks
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CKKS_TRANSFORMS_DECOMPOSE_RESCALE_H_
