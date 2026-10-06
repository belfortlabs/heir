#include "lib/Transforms/LinalgFuseLinearOps/LinalgFuseLinearOps.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Traits.h"              // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"         // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"          // from @llvm-project
#include "mlir/include/mlir/IR/AsmState.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"           // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/DialectResourceBlobManager.h"  // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"     // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"         // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"    // from @llvm-project
#include "mlir/include/mlir/Interfaces/DestinationStyleOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"     // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_LINALGFUSELINEAROPS
#include "lib/Transforms/LinalgFuseLinearOps/LinalgFuseLinearOps.h.inc"

namespace {

bool accumulatesIntoOuts(Operation* linearOp) {
  if (isa<linalg::BatchMatmulOp, linalg::BatchMatvecOp, linalg::BatchMmt4DOp,
          linalg::BatchReduceMatmulOp, linalg::BatchVecmatOp,
          linalg::ContractOp, linalg::Conv1DNcwFcwOp, linalg::Conv1DNwcWcfOp,
          linalg::Conv1DOp, linalg::Conv2DNchwFchwOp, linalg::Conv2DNchwFchwQOp,
          linalg::Conv2DNgchwFgchwOp, linalg::Conv2DNgchwGfchwOp,
          linalg::Conv2DNgchwGfchwQOp, linalg::Conv2DNhwcFhwcOp,
          linalg::Conv2DNhwcFhwcQOp, linalg::Conv2DNhwcHwcfOp,
          linalg::Conv2DNhwcHwcfQOp, linalg::Conv2DNhwgcGfhwcOp,
          linalg::Conv2DNhwgcGfhwcQOp, linalg::Conv2DOp,
          linalg::Conv3DNcdhwFcdhwOp, linalg::Conv3DNdhwcDhwcfOp,
          linalg::Conv3DNdhwcDhwcfQOp, linalg::Conv3DOp,
          linalg::DepthwiseConv1DNcwCwOp, linalg::DepthwiseConv1DNwcWcOp,
          linalg::DepthwiseConv1DNwcWcmOp, linalg::DepthwiseConv2DNchwChwOp,
          linalg::DepthwiseConv2DNhwcHwcOp, linalg::DepthwiseConv2DNhwcHwcQOp,
          linalg::DepthwiseConv2DNhwcHwcmOp, linalg::DepthwiseConv2DNhwcHwcmQOp,
          linalg::DepthwiseConv3DNcdhwCdhwOp,
          linalg::DepthwiseConv3DNdhwcDhwcOp,
          linalg::DepthwiseConv3DNdhwcDhwcmOp, linalg::DotOp, linalg::MatmulOp,
          linalg::MatvecOp, linalg::Mmt4DOp, linalg::PoolingNchwMaxOp,
          linalg::PoolingNchwSumOp, linalg::PoolingNcwMaxOp,
          linalg::PoolingNcwSumOp, linalg::PoolingNdhwcMaxOp,
          linalg::PoolingNdhwcMinOp, linalg::PoolingNdhwcSumOp,
          linalg::PoolingNhwcMaxOp, linalg::PoolingNhwcMaxUnsignedOp,
          linalg::PoolingNhwcMinOp, linalg::PoolingNhwcMinUnsignedOp,
          linalg::PoolingNhwcSumOp, linalg::PoolingNwcMaxOp,
          linalg::PoolingNwcMaxUnsignedOp, linalg::PoolingNwcMinOp,
          linalg::PoolingNwcMinUnsignedOp, linalg::PoolingNwcSumOp,
          linalg::QuantizedBatchMatmulOp, linalg::QuantizedMatmulOp,
          linalg::ReduceOp, linalg::VecmatOp>(linearOp)) {
    return true;
  }

  // Check if a generic op uses addi or addf on the outs's corresponding block
  // argument just before yielding it.
  if (auto generic = dyn_cast<linalg::GenericOp>(linearOp)) {
    if (generic.getNumDpsInits() != 1) return false;
    Block* body = generic.getBody();
    auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
    Value yieldedValue = yieldOp.getOperand(0);
    auto addOp = yieldedValue.getDefiningOp();
    if (!addOp) return false;
    if (isa<arith::AddFOp, arith::AddIOp>(addOp)) {
      Value lhs = addOp->getOperand(0);
      Value rhs = addOp->getOperand(1);
      Value outBlockArg = body->getArgument(generic.getNumDpsInputs());
      if (lhs == outBlockArg || rhs == outBlockArg) {
        return true;
      }
    }
  }

  return false;
}

// Given an op (add, sub, div, mul), try to find a linalg op whose op result is
// used by this op, populate `linearOp` with that result, and populate
// `rawOperand` with the other operand of `op`. Fail if no such linalg op can be
// found.
template <typename OpTy>
LogicalResult findLinearOpAndOperand(OpTy op, Operation*& linearOp,
                                     Value& rawOperand) {
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();

  // The fused op is rewritten in place, so its result must have no other user
  // that would see the fused scale or addend.
  auto isLinearOp = [](Operation* defOp) {
    return defOp && isa<linalg::LinalgOp>(defOp) &&
           !isa<linalg::BroadcastOp, linalg::FillOp, linalg::TransposeOp>(
               defOp) &&
           defOp->getNumResults() == 1 && defOp->getResult(0).hasOneUse();
  };

  auto* lhsOp = lhs.getDefiningOp();
  auto* rhsOp = rhs.getDefiningOp();

  if (isLinearOp(lhsOp)) {
    linearOp = lhsOp;
    rawOperand = rhs;
    return success();
  }
  if (isLinearOp(rhsOp)) {
    linearOp = rhsOp;
    rawOperand = lhs;
    return success();
  }
  return failure();
}

static void moveSliceBefore(Operation* op, Operation* target) {
  if (!op) return;
  if (op->getBlock() != target->getBlock() || op->isBeforeInBlock(target))
    return;
  for (Value operand : op->getOperands()) {
    moveSliceBefore(operand.getDefiningOp(), target);
  }
  op->moveBefore(target);
}

// Given a value, walk the use-def chain backwards to try to find a scalar SSA
// value, passing through ops that only splat/reshape a scalar as a dense
// tensor. Alternatively, if the end of the chain is a tensor type, check
// to see if it can be broadcast to a given weightsType.
FailureOr<OpFoldResult> findOriginalScalar(Value scaleVal, Type weightsType) {
  Value val = scaleVal;
  Operation* definingOp = nullptr;
  while (val.getDefiningOp() && isa<ShapedType>(val.getType())) {
    definingOp = val.getDefiningOp();

    // We hit a block argument
    if (!definingOp) break;

    if (!isa<linalg::BroadcastOp, tensor::ExpandShapeOp, tensor::SplatOp,
             arith::ConstantOp>(*definingOp))
      return failure();

    // For the supported ops, the input to traverse is always the first operand,
    // or else a constant op (with no operands) at which point we break.
    if (definingOp->getNumOperands() == 0) break;

    val = definingOp->getOperand(0);
  }

  // Case 1: we stopped at an SSA value that is already a scalar
  if (!isa<ShapedType>(val.getType())) return OpFoldResult(val);

  // Case 2: we stopped at an SSA value defined by a splatted arith.constant
  if (definingOp) {
    if (auto constantOp = dyn_cast<arith::ConstantOp>(definingOp)) {
      Attribute valueAttr = constantOp.getValue();
      if (DenseElementsAttr denseElementsAttr =
              dyn_cast<DenseElementsAttr>(valueAttr)) {
        if (denseElementsAttr.isSplat()) {
          return OpFoldResult(denseElementsAttr);
        }
      }
    }
  }

  // Case 3: we stopped at an SSA value that is a 1D tensor. This is checked
  // for broadcast compatibility by the caller.
  if (auto shapedTy = dyn_cast<ShapedType>(val.getType())) {
    if (shapedTy.getRank() == 1) return OpFoldResult(val);
  }

  return failure();
}

// If `scaleVal` is a constant of the linear op's result shape whose values only
// vary along `channelDim`, as produced by folding a broadcast per-channel scale
// (e.g. a BatchNorm's gamma / sqrt(var + eps)), return the per-channel values.
static FailureOr<DenseElementsAttr> getPerChannelValues(Value scaleVal,
                                                        int64_t channelDim) {
  DenseElementsAttr attr;
  if (!matchPattern(scaleVal, m_Constant(&attr))) return failure();
  auto type = cast<ShapedType>(attr.getType());
  int64_t numChannels = type.getDimSize(channelDim);
  int64_t stride = 1;
  for (int64_t i = channelDim + 1; i < type.getRank(); ++i)
    stride *= type.getDimSize(i);

  SmallVector<Attribute> channels(numChannels);
  int64_t index = 0;
  for (Attribute value : attr.getValues<Attribute>()) {
    Attribute& channel = channels[(index++ / stride) % numChannels];
    if (!channel)
      channel = value;
    else if (channel != value)
      return failure();
  }
  return DenseElementsAttr::get(
      RankedTensorType::get({numChannels}, type.getElementType()), channels);
}

// Fold `weights` op `scale` into a new resource constant when `weights` is a
// resource constant, which the arith folders do not read. Keeping the fused
// weights a resource lets assign_layout fold them into packed constants.
template <typename OpTy, typename T>
static Value foldIntoResourceWeights(PatternRewriter& rewriter, Location loc,
                                     DenseResourceElementsAttr weights,
                                     DenseElementsAttr scale) {
  ArrayRef<char> raw = weights.getData();
  int64_t numElements = weights.getType().getNumElements();
  if (static_cast<int64_t>(raw.size()) != numElements * sizeof(T))
    return Value();
  ArrayRef<T> input(reinterpret_cast<const T*>(raw.data()), numElements);
  SmallVector<T> result;
  result.reserve(numElements);
  auto scaleValues = scale.getValues<T>();
  for (int64_t i = 0; i < numElements; ++i) {
    T s = scale.isSplat() ? scaleValues[0] : scaleValues[i];
    if constexpr (std::is_same_v<OpTy, arith::DivFOp>)
      result.push_back(input[i] / s);
    else
      result.push_back(input[i] * s);
  }
  auto attr = DenseResourceElementsAttr::get(
      weights.getType(), "fused_weights",
      HeapAsmResourceBlob::allocateAndCopyInferAlign<T>(result));
  return arith::ConstantOp::create(rewriter, loc, attr);
}

template <typename OpTy>
static Value foldIntoResourceWeights(PatternRewriter& rewriter, Location loc,
                                     Value weights, Value scale) {
  auto weightsOp = weights.getDefiningOp<arith::ConstantOp>();
  if (!weightsOp) return Value();
  auto resource = dyn_cast<DenseResourceElementsAttr>(weightsOp.getValue());
  DenseElementsAttr scaleAttr;
  if (!resource || !matchPattern(scale, m_Constant(&scaleAttr)) ||
      scaleAttr.getType().getShape() != resource.getType().getShape())
    return Value();
  Type elementType = resource.getType().getElementType();
  if (elementType.isF32())
    return foldIntoResourceWeights<OpTy, float>(rewriter, loc, resource,
                                                scaleAttr);
  if (elementType.isF64())
    return foldIntoResourceWeights<OpTy, double>(rewriter, loc, resource,
                                                 scaleAttr);
  return Value();
}

// If `op` has a preceding LinalgOp with a compatible structure, fuse `op`
// by multiplying or dividing it by the initializer and using that as the new
// initializer.
//
// In this routine, we support fusing only a scalar multiplication or division,
// though we also support constants that are broadcast, expand-shaped, or
// dense-splatted to match the `outs` shape.
template <typename OpTy>
LogicalResult fuseScaleOrDivIntoLinearOp(PatternRewriter& rewriter, OpTy op,
                                         DataFlowSolver& solver) {
  Operation* linearOp = nullptr;
  Value scaleVal;
  if (failed(findLinearOpAndOperand(op, linearOp, scaleVal))) return failure();

  if (isSecret(scaleVal, &solver)) {
    return failure();
  }

  Value weights;
  int64_t weightOperandIdx = -1;
  int64_t matchDim = -1;
  // The dimension of the linear op's result that `matchDim` of the weights
  // produces.
  int64_t resultMatchDim = -1;
  int64_t resultRank = cast<ShapedType>(op.getType()).getRank();

  // Opt-in only specific ops where the `addend` can be fused into the
  // corresponding cleartext weights matrix.
  llvm::TypeSwitch<Operation*>(linearOp)
      .template Case<linalg::MatmulOp, linalg::VecmatOp>([&](auto op) {
        weights = op.getOperand(1);
        weightOperandIdx = 1;
        matchDim = 1;
        resultMatchDim = resultRank - 1;
      })
      .template Case<linalg::MatvecOp>([&](auto op) {
        weights = op.getOperand(0);
        weightOperandIdx = 0;
        matchDim = 0;
        resultMatchDim = 0;
      })
      .template Case<linalg::Conv2DNchwFchwOp, linalg::Conv1DNcwFcwOp>(
          [&](auto op) {
            weights = op.getOperand(1);
            weightOperandIdx = 1;
            matchDim = 0;
            resultMatchDim = 1;
          })
      .template Case<linalg::Conv2DNhwcFhwcOp>([&](auto op) {
        weights = op.getOperand(1);
        weightOperandIdx = 1;
        matchDim = 0;
        resultMatchDim = 3;
      })
      .template Case<linalg::Conv2DNhwcHwcfOp>([&](auto op) {
        weights = op.getOperand(1);
        weightOperandIdx = 1;
        matchDim = 3;
        resultMatchDim = 3;
      })
      .template Case<linalg::Conv1DNwcWcfOp>([&](auto op) {
        weights = op.getOperand(1);
        weightOperandIdx = 1;
        matchDim = 2;
        resultMatchDim = 2;
      })
      .Default([](auto) {});

  if (!weights) return failure();
  if (isSecret(weights, &solver)) {
    return failure();
  }

  auto weightsType = cast<RankedTensorType>(weights.getType());

  // In this pattern, we support fusing only a scalar multiplication or
  // division. So the `scaleVal` may need to be traced back through broadcasts
  // until a dense constant or scalar SSA value is identified.
  FailureOr<OpFoldResult> maybeScaleOfr =
      findOriginalScalar(scaleVal, weightsType);
  if (failed(maybeScaleOfr)) {
    FailureOr<DenseElementsAttr> perChannel =
        getPerChannelValues(scaleVal, resultMatchDim);
    if (failed(perChannel)) return failure();
    rewriter.setInsertionPoint(linearOp);
    maybeScaleOfr = OpFoldResult(
        arith::ConstantOp::create(rewriter, linearOp->getLoc(), *perChannel)
            .getResult());
  }
  OpFoldResult scaleOfr = *maybeScaleOfr;

  if (auto origVal = dyn_cast<Value>(scaleOfr)) {
    moveSliceBefore(origVal.getDefiningOp(), linearOp);
  }
  moveSliceBefore(scaleVal.getDefiningOp(), linearOp);
  rewriter.setInsertionPoint(linearOp);

  // Next we need to materialize the scalar as a constant, with shape matching
  // the weights we want to fuse it to.

  Value newScaleVal;
  if (auto attr = dyn_cast<Attribute>(scaleOfr)) {
    auto newDenseAttr = DenseElementsAttr::get(
        weightsType, cast<DenseElementsAttr>(attr).getSplatValue<Attribute>());
    auto newConstantOp =
        arith::ConstantOp::create(rewriter, linearOp->getLoc(), newDenseAttr);
    newScaleVal = newConstantOp.getResult();
  } else {
    Value origVal = cast<Value>(scaleOfr);
    auto origType = cast<ShapedType>(origVal.getType());
    if (origType == weightsType) {
      newScaleVal = origVal;
    } else if (origType.getRank() == 1) {
      // The added type is not broadcast compatible with the weights tensor.
      if (weightsType.getDimSize(matchDim) != origType.getDimSize(0)) {
        return failure();
      }

      SmallVector<int64_t> addedDims;
      for (int i = 0; i < weightsType.getRank(); ++i) {
        if (i != matchDim) {
          addedDims.push_back(i);
        }
      }
      // Broadcast a constant scale as a constant, so the scaled weights fold.
      DenseElementsAttr scaleAttr;
      if (matchPattern(origVal, m_Constant(&scaleAttr))) {
        SmallVector<Attribute> perChannel(scaleAttr.getValues<Attribute>());
        int64_t stride = 1;
        for (int64_t i = matchDim + 1; i < weightsType.getRank(); ++i)
          stride *= weightsType.getDimSize(i);
        SmallVector<Attribute> broadcastValues;
        broadcastValues.reserve(weightsType.getNumElements());
        for (int64_t i = 0; i < weightsType.getNumElements(); ++i)
          broadcastValues.push_back(
              perChannel[(i / stride) % weightsType.getDimSize(matchDim)]);
        newScaleVal = arith::ConstantOp::create(
            rewriter, linearOp->getLoc(),
            DenseElementsAttr::get(weightsType, broadcastValues));
      } else {
        auto emptyOp = tensor::EmptyOp::create(rewriter, linearOp->getLoc(),
                                               weightsType.getShape(),
                                               weightsType.getElementType());
        auto broadcastOp =
            linalg::BroadcastOp::create(rewriter, linearOp->getLoc(), origVal,
                                        emptyOp.getResult(), addedDims);
        newScaleVal = broadcastOp.getResults()[0];
      }
    } else {
      return failure();
    }
  }

  Value scaledWeights = foldIntoResourceWeights<OpTy>(rewriter, op.getLoc(),
                                                      weights, newScaleVal);
  if (!scaledWeights)
    scaledWeights =
        rewriter.createOrFold<OpTy>(op.getLoc(), weights, newScaleVal);

  // The linear op accumulates into its outs, so (x * W + outs) * s must also
  // scale a nonzero outs: x * (W * s) + outs * s.
  auto destStyleOp = dyn_cast<DestinationStyleOpInterface>(linearOp);
  Value scaledOuts;
  if (destStyleOp && destStyleOp.getNumDpsInits() == 1 &&
      accumulatesIntoOuts(linearOp)) {
    Value outs = destStyleOp.getDpsInitOperand(0)->get();
    if (!outs.getDefiningOp<tensor::EmptyOp>() &&
        !matchPattern(outs, m_AnyZeroFloat()))
      scaledOuts = rewriter.createOrFold<OpTy>(op.getLoc(), outs, scaleVal);
  }

  rewriter.modifyOpInPlace(linearOp, [&]() {
    linearOp->setOperand(weightOperandIdx, scaledWeights);
    if (scaledOuts) destStyleOp.setDpsInitOperand(0, scaledOuts);
  });
  rewriter.replaceOp(op, linearOp->getResults());
  return success();
}

// If `op` has a preceding LinalgOp with a single `outs` initializer, fuse `op`
// by adding or subtracting it by the initializer and using that as the new
// initializer.
template <typename OpTy>
LogicalResult fuseAddOrSubIntoLinearOp(PatternRewriter& rewriter, OpTy op,
                                       DataFlowSolver& solver) {
  Operation* linearOp = nullptr;
  Value addend;
  if (failed(findLinearOpAndOperand(op, linearOp, addend))) return failure();

  if (isSecret(addend, &solver)) {
    return failure();
  }

  auto destStyleOp = dyn_cast<DestinationStyleOpInterface>(linearOp);
  if (!destStyleOp) return failure();
  if (!accumulatesIntoOuts(linearOp)) return failure();
  if (destStyleOp.getNumDpsInits() != 1) return failure();

  auto previousOuts = destStyleOp.getDpsInitOperand(0)->get();

  // The addend and output type must match by construction: they were originally
  // added or sub'ed together by `op` which is assumed to verify.
  assert(previousOuts.getType() == addend.getType() &&
         "mismatching types for fuseAddOrSubIntoLinearOp");

  moveSliceBefore(addend.getDefiningOp(), linearOp);
  rewriter.setInsertionPoint(linearOp);
  Value newOuts;
  if (previousOuts.getDefiningOp<tensor::EmptyOp>()) {
    if constexpr (std::is_same_v<OpTy, arith::SubFOp>) {
      newOuts = rewriter.createOrFold<arith::NegFOp>(op.getLoc(), addend);
    } else {
      newOuts = addend;
    }
  } else {
    newOuts = rewriter.createOrFold<OpTy>(op.getLoc(), previousOuts, addend);
  }

  rewriter.modifyOpInPlace(
      destStyleOp, [&]() { destStyleOp.setDpsInitOperand(0, newOuts); });
  rewriter.replaceOp(op, destStyleOp->getResults());
  return success();
}

struct FuseScaleIntoLinearOp : public OpRewritePattern<arith::MulFOp> {
  FuseScaleIntoLinearOp(MLIRContext* context, DataFlowSolver& solver)
      : OpRewritePattern<arith::MulFOp>(context), solver(solver) {}
  LogicalResult matchAndRewrite(arith::MulFOp op,
                                PatternRewriter& rewriter) const override {
    return fuseScaleOrDivIntoLinearOp(rewriter, op, solver);
  }

 private:
  DataFlowSolver& solver;
};

struct FuseDivIntoLinearOp : public OpRewritePattern<arith::DivFOp> {
  FuseDivIntoLinearOp(MLIRContext* context, DataFlowSolver& solver)
      : OpRewritePattern<arith::DivFOp>(context), solver(solver) {}
  LogicalResult matchAndRewrite(arith::DivFOp op,
                                PatternRewriter& rewriter) const override {
    return fuseScaleOrDivIntoLinearOp(rewriter, op, solver);
  }

 private:
  DataFlowSolver& solver;
};

struct FuseAddIntoLinearOp : public OpRewritePattern<arith::AddFOp> {
  FuseAddIntoLinearOp(MLIRContext* context, DataFlowSolver& solver)
      : OpRewritePattern<arith::AddFOp>(context), solver(solver) {}
  LogicalResult matchAndRewrite(arith::AddFOp op,
                                PatternRewriter& rewriter) const override {
    return fuseAddOrSubIntoLinearOp(rewriter, op, solver);
  }

 private:
  DataFlowSolver& solver;
};

struct FuseSubIntoLinearOp : public OpRewritePattern<arith::SubFOp> {
  FuseSubIntoLinearOp(MLIRContext* context, DataFlowSolver& solver)
      : OpRewritePattern<arith::SubFOp>(context), solver(solver) {}
  LogicalResult matchAndRewrite(arith::SubFOp op,
                                PatternRewriter& rewriter) const override {
    return fuseAddOrSubIntoLinearOp(rewriter, op, solver);
  }

 private:
  DataFlowSolver& solver;
};

}  // namespace

struct LinalgFuseLinearOps
    : public impl::LinalgFuseLinearOpsBase<LinalgFuseLinearOps> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto module = getOperation();

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();
    if (failed(solver.initializeAndRun(module))) {
      module->emitOpError() << "Failed to run SecretnessAnalysis.\n";
      return signalPassFailure();
    }

    RewritePatternSet patterns(context);
    patterns.add<FuseScaleIntoLinearOp, FuseDivIntoLinearOp,
                 FuseAddIntoLinearOp, FuseSubIntoLinearOp>(context, solver);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createLinalgFuseLinearOpsPass() {
  return std::make_unique<LinalgFuseLinearOps>();
}

}  // namespace heir
}  // namespace mlir
