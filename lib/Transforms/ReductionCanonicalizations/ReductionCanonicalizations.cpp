#include "lib/Transforms/ReductionCanonicalizations/ReductionCanonicalizations.h"

#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/AffineMap.h>

#include <cassert>
#include <functional>
#include <utility>

#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"      // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"      // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/LinalgInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"              // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "reduction-canonicalizations"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_REDUCTIONCANONICALIZATIONS
#include "lib/Transforms/ReductionCanonicalizations/ReductionCanonicalizations.h.inc"

struct GenericToReduce : public OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  GenericToReduce(MLIRContext* context)
      : OpRewritePattern<mlir::linalg::GenericOp>(context) {}

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::linalg::GenericOp genericOp,
                                PatternRewriter& rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "[generic-to-reduce] Trying to rewrite: ");
    LLVM_DEBUG(genericOp.print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n");

    // make sure we have equal number of input and output
    if (genericOp.getInputs().size() != genericOp.getDpsInits().size()) {
      return rewriter.notifyMatchFailure(
          genericOp,
          "genericOp with mismatched input/output counts cannot become a "
          "linalg.reduce");
    }

    // make sure we need to reduce and collect dimensions
    mlir::SmallVector<int64_t> dimensions = {};
    for (int i = 0; i < genericOp.getIteratorTypesArray().size(); i++) {
      auto itType = genericOp.getIteratorTypesArray()[i];
      if (itType == utils::IteratorType::reduction) {
        dimensions.push_back(i);
      } else if (itType != utils::IteratorType::parallel) {
        return rewriter.notifyMatchFailure(
            genericOp,
            "genericOp with non-reduction, non-parallel iterator type not "
            "supported");
      }
    }

    if (dimensions.empty()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[generic-to-reduce] No reduction dimensions found\n");
      return rewriter.notifyMatchFailure(
          genericOp, "genericOp with no reduction dimensions not supported");
    }

    // check all affine maps
    for (int i = 0; i < genericOp.getIndexingMaps().size(); i++) {
      if (i < genericOp.getInputs().size()) {  // input maps
        if (!llvm::cast<AffineMapAttr>(genericOp.getIndexingMaps()[i])
                 .getAffineMap()
                 .isIdentity()) {
          return rewriter.notifyMatchFailure(
              genericOp,
              "genericOp with incompatible input dimensions not supported");
        }
      } else {  // output maps
        if (!checkOutputMap(
                llvm::cast<AffineMapAttr>(genericOp.getIndexingMaps()[i])
                    .getAffineMap(),
                dimensions)) {
          LLVM_DEBUG(llvm::dbgs() << "[generic-to-reduce] Output map " << i
                                  << " is not right\n");
          return rewriter.notifyMatchFailure(
              genericOp,
              "genericOp with incompatible output dimensions not supported");
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "[generic-to-reduce] Output map " << i << " is right\n");
      }
    }

    // make a linalg.reduceOp to replace the linalg.genericOp
    mlir::linalg::ReduceOp reduce = mlir::linalg::ReduceOp::create(
        rewriter, genericOp.getLoc(), TypeRange(genericOp.getDpsInits()),
        genericOp.getInputs(), SmallVector<Value>(genericOp.getDpsInits()),
        dimensions);

    // generic ops have 1 block
    rewriter.cloneRegionBefore(genericOp.getRegion(), reduce.getRegion(),
                               reduce.getRegion().end());

    rewriter.replaceAllOpUsesWith(genericOp, reduce.getResults());
    rewriter.eraseOp(genericOp);

    return success();
  }

 private:
  // This function assures that the output map is an identity map with the
  // reduction dimensions removed. Ex: (d0, d1) -> (d0) or (d0) -> ().
  bool checkOutputMap(const AffineMap map,
                      const mlir::SmallVector<int64_t> dimensions) const {
    // check total size
    LLVM_DEBUG(llvm::dbgs() << "# of reduction dims: " << dimensions.size()
                            << " # of map dims: " << map.getNumDims()
                            << " # of results: " << map.getNumResults());
    if (map.getNumResults() != map.getNumDims() - dimensions.size()) {
      LLVM_DEBUG(llvm::dbgs() << "not right # of map dims");
      return false;
    }
    // Example pass: (d0) -> ()
    if (map.getNumResults() == 0) {
      return true;
    }
    // (d0, d1) -> (d0)
    int resultIndex =
        0;  // output map index, i is just the index in the map inputs
    ArrayRef<AffineExpr> results = map.getResults();
    for (unsigned i = 0, numDims = map.getNumResults(); i < numDims; ++i) {
      bool isReduced = find(dimensions, i) != dimensions.end();
      if (isReduced) {  // skip this on the output dimension
        continue;
      }
      // based on isIdentity
      auto expr = dyn_cast<AffineDimExpr>(results[resultIndex]);
      if (!expr || expr.getPosition() != i) {
        LLVM_DEBUG(llvm::dbgs()
                   << "map result " << expr.getPosition()
                   << " is not an affine dim expr with expected position "
                   << resultIndex);
        return false;
      }
      resultIndex++;
    }
    return true;
  }
};

// Splits a sum or product over several dimensions into a chain of
// single-dimension reductions, innermost dimension first, which is what the
// ciphertext-semantics kernel for linalg.reduce supports. Every reduction but
// the last starts from the operation's identity; the last one takes the
// original init.
struct SplitMultiDimReduce : public OpRewritePattern<linalg::ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ReduceOp reduceOp,
                                PatternRewriter& rewriter) const override {
    ArrayRef<int64_t> dimensions = reduceOp.getDimensions();
    if (dimensions.size() < 2 || !reduceOp.isSingleInputOutput())
      return rewriter.notifyMatchFailure(reduceOp,
                                         "not a multi-dimension reduction");

    Block* body = reduceOp.getBlock();
    if (body->getOperations().size() != 2)
      return rewriter.notifyMatchFailure(reduceOp, "body is not a single op");
    Operation* combiner = &body->front();
    auto inputType =
        dyn_cast<RankedTensorType>(reduceOp.getInputs()[0].getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(reduceOp, "input is not a tensor");
    Type elementType = inputType.getElementType();

    TypedAttr identity;
    if (isa<arith::AddFOp>(combiner))
      identity = rewriter.getFloatAttr(elementType, 0.0);
    else if (isa<arith::MulFOp>(combiner))
      identity = rewriter.getFloatAttr(elementType, 1.0);
    else if (isa<arith::AddIOp>(combiner))
      identity = rewriter.getIntegerAttr(elementType, 0);
    else if (isa<arith::MulIOp>(combiner))
      identity = rewriter.getIntegerAttr(elementType, 1);
    else
      return rewriter.notifyMatchFailure(reduceOp,
                                         "combiner has no known identity");

    // Reducing the highest dimension first leaves the indices of the
    // remaining ones unchanged.
    SmallVector<int64_t> order(dimensions);
    llvm::sort(order, std::greater<int64_t>());

    Value current = reduceOp.getInputs()[0];
    SmallVector<int64_t> shape(inputType.getShape());
    for (auto [i, dim] : llvm::enumerate(order)) {
      shape.erase(shape.begin() + dim);
      Value init;
      if (i + 1 == order.size()) {
        init = reduceOp.getInits()[0];
      } else {
        auto type = RankedTensorType::get(shape, elementType);
        init =
            arith::ConstantOp::create(rewriter, reduceOp.getLoc(), type,
                                      DenseElementsAttr::get(type, identity));
      }
      auto step = linalg::ReduceOp::create(
          rewriter, reduceOp.getLoc(), TypeRange{init.getType()},
          ValueRange{current}, ValueRange{init}, ArrayRef<int64_t>{dim});
      rewriter.cloneRegionBefore(reduceOp.getRegion(), step.getRegion(),
                                 step.getRegion().end());
      current = step.getResult(0);
    }
    rewriter.replaceOp(reduceOp, current);
    return success();
  }
};

struct ReductionCanonicalizations
    : public impl::ReductionCanonicalizationsBase<ReductionCanonicalizations> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<GenericToReduce>(context);

    walkAndApplyPatterns(module, std::move(patterns));

    RewritePatternSet splitPatterns(context);
    splitPatterns.add<SplitMultiDimReduce>(context);
    walkAndApplyPatterns(module, std::move(splitPatterns));
  }
};

}  // namespace heir
}  // namespace mlir
