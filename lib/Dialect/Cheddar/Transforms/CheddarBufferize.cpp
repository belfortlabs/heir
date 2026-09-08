#include "lib/Dialect/Cheddar/Transforms/CheddarBufferize.h"

#include "mlir/include/mlir/Dialect/Bufferization/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"   // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"      // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {

#define GEN_PASS_DEF_CHEDDARELIDEOUTPARAMCOPIES
#include "lib/Dialect/Cheddar/Transforms/Passes.h.inc"

namespace {

// `%tmp = memref.alloc(); ...; memref.copy %tmp, %out` where `%out` is an
// otherwise unused out-param: write into `%out` from the start. This is the
// case `buffer-results-to-out-params=hoist-static-allocs` misses, e.g. a
// function returning the result of a call.
struct ElideOutParamCopy : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter& rewriter) const override {
    auto dest = dyn_cast<BlockArgument>(copy.getTarget());
    if (!dest || !dest.hasOneUse()) return failure();
    auto func = dyn_cast<func::FuncOp>(dest.getOwner()->getParentOp());
    if (!func || !func.getArgAttr(dest.getArgNumber(), "bufferize.result"))
      return failure();
    auto alloc = copy.getSource().getDefiningOp<memref::AllocOp>();
    if (!alloc || alloc.getType() != dest.getType()) return failure();
    // The copy must be the last use of the temporary.
    for (Operation* user : alloc->getUsers()) {
      if (user == copy) continue;
      Operation* ancestor = copy->getBlock()->findAncestorOpInBlock(*user);
      if (!ancestor || !ancestor->isBeforeInBlock(copy)) return failure();
    }
    rewriter.replaceAllUsesWith(alloc.getResult(), dest);
    rewriter.eraseOp(copy);
    rewriter.eraseOp(alloc);
    return success();
  }
};

struct CheddarElideOutParamCopies
    : public impl::CheddarElideOutParamCopiesBase<CheddarElideOutParamCopies> {
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ElideOutParamCopy>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

void buildCheddarBufferizationPipeline(OpPassManager& pm) {
  pm.addPass(bufferization::createEmptyTensorEliminationPass());
  bufferization::OneShotBufferizePassOptions oneShot;
  oneShot.bufferizeFunctionBoundaries = true;
  oneShot.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(bufferization::createOneShotBufferizePass(oneShot));
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  bufferization::DropEquivalentBufferResultsPassOptions dropEquivalent;
  dropEquivalent.modifyPublicFunctions = true;
  pm.addPass(
      bufferization::createDropEquivalentBufferResultsPass(dropEquivalent));
  bufferization::BufferResultsToOutParamsPassOptions outParams;
  outParams.hoistStaticAllocs = true;
  outParams.addResultAttribute = true;
  outParams.modifyPublicFunctions = true;
  pm.addPass(bufferization::createBufferResultsToOutParamsPass(outParams));
  pm.addNestedPass<func::FuncOp>(createCheddarElideOutParamCopies());
  pm.addPass(createCanonicalizerPass());
}

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
