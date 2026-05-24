#include "lib/Dialect/CKKS/Transforms/DecomposeRescale.h"

#include <utility>

#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"

namespace mlir {
namespace heir {
namespace ckks {

#define GEN_PASS_DEF_DECOMPOSERESCALE
#include "lib/Dialect/CKKS/Transforms/Passes.h.inc"

LogicalResult DecomposeRescalePattern::matchAndRewrite(RescaleOp op,
                            PatternRewriter& rewriter) const {
    // TODO(bence): implement
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
}

struct DecomposeRescale
    : impl::DecomposeRescaleBase<DecomposeRescale> {
  using impl::DecomposeRescaleBase<DecomposeRescale>::DecomposeRescaleBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<DecomposeRescalePattern>(context);
    (void)walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace ckks
} // namespace heir
} // namespace mlir
