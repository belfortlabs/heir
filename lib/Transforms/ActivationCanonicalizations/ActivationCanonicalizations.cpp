#include "lib/Transforms/ActivationCanonicalizations/ActivationCanonicalizations.h"

#include <utility>

#include "lib/Dialect/MathExt/IR/MathExtOps.h"
#include "llvm/include/llvm/ADT/APFloat.h"               // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ACTIVATIONCANONICALIZATIONS
#include "lib/Transforms/ActivationCanonicalizations/ActivationCanonicalizations.h.inc"

static bool IsOne(mlir::Attribute attr) {
  mlir::FloatType floatTy;
  llvm::APFloat floatVal(0.0);

  if (auto splattr = mlir::dyn_cast_or_null<mlir::SplatElementsAttr>(attr)) {
    floatTy = mlir::dyn_cast_or_null<mlir::FloatType>(splattr.getElementType());
    floatVal = splattr.getValues<llvm::APFloat>()[0];
  } else if (auto floatAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(attr)) {
    floatTy = mlir::dyn_cast_or_null<mlir::FloatType>(floatAttr.getType());
    floatVal = floatAttr.getValue();
  }

  if (!floatTy) return false;
  auto one =
      llvm::APFloat::getOne(floatTy.getFloatSemantics(), /*Negative=*/false);
  return floatVal == one;
}

// Kept inside a namespace because it generates a function called
// populateWithGenerated, which can conflict with other generated patterns.
#include "lib/Transforms/ActivationCanonicalizations/Rewrites.cpp.inc"

// select(a > c, a, c) = max(a, c) for floats. This is the C++ equivalent of
// the DRR `SelectGreaterThanEqualFloat` pattern (now removed from Rewrites.td),
// extended to forward `domain_lower`/`domain_upper`: a torch ReLU imports as a
// `linalg.generic` carrying those attrs on the generic op, with `cmpf+select`
// in its body. A DRR pattern matching the inner select has no access to the
// enclosing generic, so the domain would be lost and HEIR's
// PolynomialApproximation / ReluViaCompositeSign would fall back to [-1, 1].
// Here we copy the attrs from the parent generic onto the new `arith.maximumf`
// (which is what PolynomialApproximation reads).
struct SelectGreaterThanEqualFloatPattern
    : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter& rewriter) const override {
    auto cmpOp = op.getCondition().getDefiningOp<arith::CmpFOp>();
    if (!cmpOp)
      return rewriter.notifyMatchFailure(op, "condition is not arith.cmpf");

    auto pred = cmpOp.getPredicate();
    if (pred != arith::CmpFPredicate::UGT && pred != arith::CmpFPredicate::UGE)
      return rewriter.notifyMatchFailure(op, "predicate is not ugt/uge");

    // Must be the ReLU/max shape: select(a >? c, a, c).
    if (cmpOp.getLhs() != op.getTrueValue() ||
        cmpOp.getRhs() != op.getFalseValue())
      return rewriter.notifyMatchFailure(op,
                                         "operands are not select(a>c,a,c)");

    auto maxOp =
        arith::MaximumFOp::create(rewriter, op.getLoc(), op.getTrueValue(),
                                  op.getFalseValue(), cmpOp.getFastmathAttr());

    // Forward polynomial-approximation domain bounds from an enclosing
    // linalg.generic (where torch-mlir's importer attaches them) so they land
    // on the op PolynomialApproximation actually reads. We then strip them
    // from the generic: leaving the bounds on BOTH the generic and the lifted
    // maximumf makes a later activation-lifting pass merge two `domain_lower`
    // entries into one attribute dictionary, which trips
    // DictionaryAttr's uniqueness assertion.
    if (auto generic = dyn_cast<linalg::GenericOp>(op->getParentOp())) {
      auto lo = generic->getAttr("domain_lower");
      auto hi = generic->getAttr("domain_upper");
      if (lo) maxOp->setAttr("domain_lower", lo);
      if (hi) maxOp->setAttr("domain_upper", hi);
      if (lo || hi) {
        rewriter.modifyOpInPlace(generic, [&]() {
          generic->removeAttr("domain_lower");
          generic->removeAttr("domain_upper");
        });
      }
    }

    rewriter.replaceOp(op, maxOp.getResult());
    return success();
  }
};

struct ActivationCanonicalizations
    : impl::ActivationCanonicalizationsBase<ActivationCanonicalizations> {
  using ActivationCanonicalizationsBase::ActivationCanonicalizationsBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    populateWithGenerated(patterns);
    patterns.add<SelectGreaterThanEqualFloatPattern>(context);

    (void)walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
