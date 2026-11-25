#include "lib/Transforms/LinearKernel/ReluNormalization.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/StringExtras.h"          // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"               // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_RELUCALIBRATION
#define GEN_PASS_DEF_RELUNORMALIZATION
#include "lib/Transforms/LinearKernel/ReluNormalization.h.inc"

namespace {

struct Relu {
  arith::MaximumFOp op;
  Value input;
  Value zero;
  RankedTensorType type;
  int64_t channels;
};

// The ReLUs of a function in program order: `arith.maximumf` of a statically
// shaped f32 tensor and a zero constant.
std::vector<Relu> collectRelus(func::FuncOp func) {
  std::vector<Relu> relus;
  func.walk<WalkOrder::PreOrder>([&](arith::MaximumFOp op) {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type || !type.hasStaticShape() || !type.getElementType().isF32())
      return;
    Value input, zero;
    if (matchPattern(op.getRhs(), m_AnyZeroFloat())) {
      input = op.getLhs();
      zero = op.getRhs();
    } else if (matchPattern(op.getLhs(), m_AnyZeroFloat())) {
      input = op.getRhs();
      zero = op.getLhs();
    } else {
      return;
    }
    int64_t channels = type.getRank() == 4 ? type.getDimSize(1) : 1;
    relus.push_back({op, input, zero, type, channels});
  });
  return relus;
}

Value zeros(OpBuilder& b, Location loc, RankedTensorType type) {
  return arith::ConstantOp::create(b, loc, type, b.getZeroAttr(type));
}

}  // namespace

struct ReluCalibration : impl::ReluCalibrationBase<ReluCalibration> {
  using ReluCalibrationBase::ReluCalibrationBase;

  void runOnOperation() override {
    for (auto func : getOperation().getOps<func::FuncOp>()) {
      if (func.isExternal()) continue;
      std::vector<Relu> relus = collectRelus(func);
      if (relus.empty()) continue;
      func::ReturnOp returnOp;
      func.walk([&](func::ReturnOp op) { returnOp = op; });
      if (!returnOp) continue;

      OpBuilder b(returnOp);
      Location loc = returnOp.getLoc();
      Type f32 = b.getF32Type();
      int64_t total = 0;
      for (const Relu& relu : relus) total += 2 * relu.channels;
      Value stats = zeros(b, loc, RankedTensorType::get({total}, f32));

      int64_t offset = 0;
      for (Relu relu : relus) {
        Value x = relu.input;
        Value negated = arith::NegFOp::create(b, loc, x);
        Value magnitude = arith::MaximumFOp::create(b, loc, x, negated);
        int64_t rank = relu.type.getRank();
        bool perChannel = rank == 4;
        SmallVector<int64_t> dimensions;
        for (int64_t i = 0; i < rank; ++i)
          if (!(perChannel && i == 1)) dimensions.push_back(i);
        auto reducedType = perChannel
                               ? RankedTensorType::get({relu.channels}, f32)
                               : RankedTensorType::get({}, f32);
        auto reduce = [&](bool isMax) -> Value {
          Value result =
              linalg::ReduceOp::create(
                  b, loc, ValueRange{magnitude},
                  ValueRange{zeros(b, loc, reducedType)}, dimensions,
                  [&](OpBuilder& nb, Location nloc, ValueRange args) {
                    Value v = isMax ? arith::MaximumFOp::create(
                                          nb, nloc, args[0], args[1])
                                          .getResult()
                                    : arith::AddFOp::create(nb, nloc, args[0],
                                                            args[1])
                                          .getResult();
                    linalg::YieldOp::create(nb, nloc, v);
                  })
                  .getResult(0);
          if (!perChannel) {
            Value scalar =
                tensor::ExtractOp::create(b, loc, result, ValueRange{});
            result = tensor::FromElementsOp::create(
                b, loc, RankedTensorType::get({1}, f32), ValueRange{scalar});
          }
          return result;
        };
        auto vectorType = RankedTensorType::get({relu.channels}, f32);
        Value maxima = reduce(/*isMax=*/true);
        Value sums = reduce(/*isMax=*/false);
        double count = static_cast<double>(relu.type.getNumElements()) /
                       static_cast<double>(relu.channels);
        Value countValue = arith::ConstantOp::create(
            b, loc, vectorType,
            DenseElementsAttr::get(vectorType, static_cast<float>(count)));
        Value means = arith::DivFOp::create(b, loc, sums, countValue);
        auto insert = [&](Value part, int64_t at) {
          stats = tensor::InsertSliceOp::create(
              b, loc, part, stats, {b.getIndexAttr(at)},
              {b.getIndexAttr(relu.channels)}, {b.getIndexAttr(1)});
        };
        insert(maxima, offset);
        insert(means, total / 2 + offset);
        offset += relu.channels;
      }

      returnOp->setOperands({stats});
      func.setType(FunctionType::get(func.getContext(),
                                     func.getFunctionType().getInputs(),
                                     {stats.getType()}));
    }
  }
};

struct ReluNormalization : impl::ReluNormalizationBase<ReluNormalization> {
  using ReluNormalizationBase::ReluNormalizationBase;

  void runOnOperation() override {
    std::vector<double> stats;
    {
      std::ifstream file(calibrationFile);
      double value;
      while (file >> value) stats.push_back(value);
    }
    // Worst-case errors of the rungs on [-1, 1], cheapest rung first.
    const char* names[] = {"deg5", "deg15", "deg31", "deg63", "composite"};
    const int degrees[] = {5, 15, 31, 63, 0};
    std::vector<double> errors;
    {
      SmallVector<StringRef> parts;
      StringRef(rungErrors).split(parts, ':', -1, /*KeepEmpty=*/false);
      for (StringRef part : parts) {
        double value = 0;
        if (part.getAsDouble(value)) {
          getOperation()->emitError() << "bad rung-errors entry " << part;
          return signalPassFailure();
        }
        errors.push_back(value);
      }
    }
    int fixedRung = -1;
    for (int i = 0; i < 5; ++i)
      if (rung == names[i]) fixedRung = i;
    if (fixedRung < 0 && (rung != "auto" || errors.size() != 5)) {
      getOperation()->emitError()
          << "rung must be one of deg5, deg15, deg31, deg63, composite, or "
             "auto with five rung-errors";
      return signalPassFailure();
    }
    bool perChannelMode = mode == "per-channel";
    if (!perChannelMode && mode != "per-tensor") {
      getOperation()->emitError() << "mode must be per-channel or per-tensor";
      return signalPassFailure();
    }

    size_t position = 0;
    int index = 0;
    for (auto func : getOperation().getOps<func::FuncOp>()) {
      if (func.isExternal()) continue;
      for (Relu relu : collectRelus(func)) {
        int64_t channels = relu.channels;
        size_t half = stats.size() / 2;
        if (position + channels > half) {
          relu.op->emitError() << "calibration file has too few entries";
          return signalPassFailure();
        }
        const double* maxima = &stats[position];
        const double* means = &stats[half + position];
        position += channels;

        double tensorMax = 0, tensorMean = 0;
        for (int64_t c = 0; c < channels; ++c) {
          tensorMax = std::max(tensorMax, maxima[c]);
          tensorMean += means[c] / channels;
        }
        if (!(tensorMax > 0)) tensorMax = 1.0 / margin;
        // A channel that was zero on every calibration input gets the bound
        // of the tensor.
        std::vector<double> scales(channels);
        std::vector<double> ratios;
        for (int64_t c = 0; c < channels; ++c) {
          bool live = maxima[c] > 0;
          scales[c] = margin * (perChannelMode && live ? maxima[c] : tensorMax);
          if (perChannelMode && live) ratios.push_back(means[c] / scales[c]);
        }
        double ratio;
        if (perChannelMode && !ratios.empty()) {
          std::sort(ratios.begin(), ratios.end());
          ratio = ratios[ratios.size() / 2];
        } else {
          ratio = tensorMean / (margin * tensorMax);
        }

        int chosen = fixedRung;
        if (chosen < 0) {
          chosen = 4;
          for (int i = 0; i < 5; ++i) {
            if (errors[i] <= tau * ratio) {
              chosen = i;
              break;
            }
          }
        }

        OpBuilder b(relu.op);
        Location loc = relu.op.getLoc();
        RankedTensorType type = relu.type;
        int64_t elements = type.getNumElements();
        int64_t perChannelElements = 1;
        if (type.getRank() == 4)
          perChannelElements = type.getDimSize(2) * type.getDimSize(3);
        std::vector<float> scaleValues(elements), inverseValues(elements);
        for (int64_t i = 0; i < elements; ++i) {
          int64_t c = (i / perChannelElements) % channels;
          scaleValues[i] = static_cast<float>(scales[c]);
          inverseValues[i] = static_cast<float>(1.0 / scales[c]);
        }
        Value scale = arith::ConstantOp::create(
            b, loc, type,
            DenseElementsAttr::get(type, ArrayRef<float>(scaleValues)));
        Value inverse = arith::ConstantOp::create(
            b, loc, type,
            DenseElementsAttr::get(type, ArrayRef<float>(inverseValues)));
        Value scaled = arith::MulFOp::create(b, loc, relu.input, inverse);
        auto unitRelu = arith::MaximumFOp::create(b, loc, scaled, relu.zero);
        unitRelu->setAttr("domain_lower", b.getF64FloatAttr(-1.0));
        unitRelu->setAttr("domain_upper", b.getF64FloatAttr(1.0));
        if (chosen == 4) {
          unitRelu->setAttr("relu_method", b.getStringAttr("composite"));
        } else {
          unitRelu->setAttr("relu_method", b.getStringAttr("polynomial"));
          unitRelu->setAttr("degree", b.getI32IntegerAttr(degrees[chosen]));
        }
        Value result =
            arith::MulFOp::create(b, loc, unitRelu.getResult(), scale);
        relu.op.getResult().replaceAllUsesWith(result);
        relu.op.erase();

        double minScale = *std::min_element(scales.begin(), scales.end());
        double maxScale = *std::max_element(scales.begin(), scales.end());
        llvm::errs() << "relu-normalization\t" << index++ << "\t" << channels
                     << "\t" << ratio << "\t" << minScale << "\t" << maxScale
                     << "\t" << names[chosen] << "\n";
      }
    }
    if (2 * position != stats.size()) {
      getOperation()->emitError()
          << "calibration file has " << stats.size() << " entries, the module "
          << "needs " << 2 * position;
      return signalPassFailure();
    }
  }
};

}  // namespace heir
}  // namespace mlir
