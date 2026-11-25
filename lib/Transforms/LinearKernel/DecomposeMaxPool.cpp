#include "lib/Transforms/LinearKernel/DecomposeMaxPool.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "llvm/include/llvm/ADT/APFloat.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"    // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"      // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_DECOMPOSEMAXPOOL
#include "lib/Transforms/LinearKernel/DecomposeMaxPool.h.inc"

namespace {

constexpr StringRef kDomainLower = "domain_lower";
constexpr StringRef kDomainUpper = "domain_upper";

// True if `value` is a float constant that never wins a maximum.
bool isMaxIdentity(Value value) {
  APFloat constant(0.0);
  if (!value || !matchPattern(value, m_ConstantFloat(&constant))) return false;
  if (!constant.isNegative()) return false;
  return constant.isInfinity() || constant.isLargest();
}

// True if a tensor holds only the identity of the maximum, or has no defined
// contents.
bool isMaxIdentityTensor(Value value) {
  if (value.getDefiningOp<tensor::EmptyOp>()) return true;
  if (auto fill = value.getDefiningOp<linalg::FillOp>()) {
    return isMaxIdentity(fill.getInputs()[0]);
  }
  return false;
}

// One spatial dimension of the pooling.
struct Axis {
  int64_t inputSize;
  int64_t outputSize;
  int64_t kernel;
  int64_t stride;
  int64_t dilation;
  int64_t padLow;

  // The input index read by tap `tap` of window `window`; outside
  // `[0, inputSize)` it is padding.
  int64_t read(int64_t window, int64_t tap) const {
    return stride * window + dilation * tap - padLow;
  }
  bool readsPadding(int64_t window, int64_t tap) const {
    int64_t index = read(window, tap);
    return index < 0 || index >= inputSize;
  }
  bool tapReadsPadding(int64_t tap) const {
    for (int64_t window = 0; window < outputSize; ++window) {
      if (readsPadding(window, tap)) return true;
    }
    return false;
  }
};

struct DecomposeMaxPoolPattern
    : public OpRewritePattern<linalg::PoolingNchwMaxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::PoolingNchwMaxOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getDpsInputs()[0];
    Value window = op.getDpsInputs()[1];
    Value init = op.getDpsInits()[0];
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto windowType = dyn_cast<RankedTensorType>(window.getType());
    auto resultType = dyn_cast<RankedTensorType>(init.getType());
    if (!inputType || !windowType || !resultType ||
        !inputType.hasStaticShape() || !windowType.hasStaticShape() ||
        !resultType.hasStaticShape() ||
        !isa<FloatType>(resultType.getElementType())) {
      return rewriter.notifyMatchFailure(
          op, "requires static shapes and a float element type");
    }
    if (!isMaxIdentityTensor(init)) {
      return rewriter.notifyMatchFailure(
          op, "the initial value is not the identity of the maximum");
    }

    // A pad with the identity of the maximum is replaced by a zero pad.
    Value source = input;
    SmallVector<int64_t> low(4, 0);
    auto pad = input.getDefiningOp<tensor::PadOp>();
    bool foldPad = pad && pad.getLow().empty() && pad.getHigh().empty() &&
                   pad.getSourceType().hasStaticShape() &&
                   isMaxIdentity(pad.getConstantPaddingValue()) &&
                   pad.getStaticLow()[0] == 0 && pad.getStaticLow()[1] == 0 &&
                   pad.getStaticHigh()[0] == 0 && pad.getStaticHigh()[1] == 0;
    if (foldPad) {
      source = pad.getSource();
      low.assign(pad.getStaticLow().begin(), pad.getStaticLow().end());
    }
    ArrayRef<int64_t> sourceShape =
        cast<RankedTensorType>(source.getType()).getShape();

    auto strides = op.getStrides().getValues<int64_t>();
    auto dilations = op.getDilations().getValues<int64_t>();
    Axis axes[2];
    for (int i = 0; i < 2; ++i) {
      axes[i] = {sourceShape[2 + i],
                 resultType.getShape()[2 + i],
                 windowType.getShape()[i],
                 strides[i],
                 dilations[i],
                 low[2 + i]};
    }

    // A tap that never reads padding stands in for the others where they do.
    int64_t anchor[2] = {-1, -1};
    for (int i = 0; i < 2; ++i) {
      for (int64_t tap = 0; tap < axes[i].kernel && anchor[i] < 0; ++tap) {
        if (!axes[i].tapReadsPadding(tap)) anchor[i] = tap;
      }
      if (anchor[i] < 0) {
        return rewriter.notifyMatchFailure(
            op, "every window position reads padding at some window");
      }
    }

    Type elementType = resultType.getElementType();
    Value readFrom = source;
    if (foldPad) {
      Value zero = arith::ConstantOp::create(
          rewriter, loc, rewriter.getFloatAttr(elementType, 0.0));
      readFrom = tensor::PadOp::create(rewriter, loc, pad.getResultType(),
                                       source, pad.getMixedLowPad(),
                                       pad.getMixedHighPad(), zero);
    }

    AffineMap identity = rewriter.getMultiDimIdentityMap(4);
    SmallVector<utils::IteratorType> iterators(4,
                                               utils::IteratorType::parallel);
    auto emitTap = [&](int64_t tapH, int64_t tapW) -> Value {
      AffineExpr n, c, h, w;
      bindDims(rewriter.getContext(), n, c, h, w);
      AffineMap read =
          AffineMap::get(4, 0,
                         {n, c, h * axes[0].stride + tapH * axes[0].dilation,
                          w * axes[1].stride + tapW * axes[1].dilation},
                         rewriter.getContext());
      Value empty = tensor::EmptyOp::create(rewriter, loc,
                                            resultType.getShape(), elementType);
      return linalg::GenericOp::create(
                 rewriter, loc, TypeRange{resultType}, ValueRange{readFrom},
                 ValueRange{empty}, ArrayRef<AffineMap>{read, identity},
                 iterators,
                 [](OpBuilder& b, Location loc, ValueRange args) {
                   linalg::YieldOp::create(b, loc, args[0]);
                 })
          .getResult(0);
    };

    Value anchorTap = emitTap(anchor[0], anchor[1]);
    std::vector<Value> taps;
    for (int64_t tapH = 0; tapH < axes[0].kernel; ++tapH) {
      for (int64_t tapW = 0; tapW < axes[1].kernel; ++tapW) {
        if (tapH == anchor[0] && tapW == anchor[1]) {
          taps.push_back(anchorTap);
          continue;
        }
        Value tap = emitTap(tapH, tapW);
        if (axes[0].tapReadsPadding(tapH) || axes[1].tapReadsPadding(tapW)) {
          // Where this tap read padding it is zero; add the anchor there.
          std::vector<APFloat> mask;
          const llvm::fltSemantics& semantics =
              cast<FloatType>(elementType).getFloatSemantics();
          for (int64_t i = 0; i < axes[0].outputSize; ++i) {
            for (int64_t j = 0; j < axes[1].outputSize; ++j) {
              bool padding = axes[0].readsPadding(i, tapH) ||
                             axes[1].readsPadding(j, tapW);
              APFloat value(padding ? 1.0 : 0.0);
              bool losesInfo;
              value.convert(semantics, APFloat::rmNearestTiesToEven,
                            &losesInfo);
              mask.push_back(value);
            }
          }
          auto maskType = RankedTensorType::get(
              {axes[0].outputSize, axes[1].outputSize}, elementType);
          Value maskValue = arith::ConstantOp::create(
              rewriter, loc, DenseElementsAttr::get(maskType, mask));
          Value empty = tensor::EmptyOp::create(
              rewriter, loc, resultType.getShape(), elementType);
          Value broadcast =
              linalg::BroadcastOp::create(rewriter, loc, maskValue, empty,
                                          ArrayRef<int64_t>{0, 1})
                  ->getResult(0);
          Value standIn =
              arith::MulFOp::create(rewriter, loc, anchorTap, broadcast);
          tap = arith::AddFOp::create(rewriter, loc, tap, standIn);
        }
        taps.push_back(tap);
      }
    }

    // max(a, b) = b + max(a - b, 0).
    Value zeros = arith::ConstantOp::create(rewriter, loc,
                                            rewriter.getZeroAttr(resultType));
    auto lower = dyn_cast_or_null<FloatAttr>(op->getAttr(kDomainLower));
    auto upper = dyn_cast_or_null<FloatAttr>(op->getAttr(kDomainUpper));
    while (taps.size() > 1) {
      std::vector<Value> next;
      for (size_t i = 0; i + 1 < taps.size(); i += 2) {
        Value diff = arith::SubFOp::create(rewriter, loc, taps[i], taps[i + 1]);
        auto relu = arith::MaximumFOp::create(rewriter, loc, diff, zeros);
        if (lower && upper) {
          double range = upper.getValueAsDouble() - lower.getValueAsDouble();
          relu->setAttr(kDomainLower, rewriter.getF64FloatAttr(-range));
          relu->setAttr(kDomainUpper, rewriter.getF64FloatAttr(range));
        }
        next.push_back(arith::AddFOp::create(rewriter, loc, taps[i + 1], relu));
      }
      if (taps.size() % 2 == 1) next.push_back(taps.back());
      taps = std::move(next);
    }
    rewriter.replaceOp(op, taps.front());
    return success();
  }
};

}  // namespace

struct DecomposeMaxPool : impl::DecomposeMaxPoolBase<DecomposeMaxPool> {
  using DecomposeMaxPoolBase::DecomposeMaxPoolBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeMaxPoolPattern>(&getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace heir
}  // namespace mlir
