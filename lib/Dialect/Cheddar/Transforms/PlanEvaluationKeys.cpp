#include "lib/Dialect/Cheddar/Transforms/PlanEvaluationKeys.h"

#include <cstdint>
#include <exception>
#include <span>
#include <utility>
#include <vector>

#include "core/EvkRequest.h"                             // from @cyclops
#include "core/Parameter.h"                              // from @cyclops
#include "extension/boot/BootKeyPlanner.h"               // from @cyclops
#include "extension/boot/BootParameter.h"                // from @cyclops
#include "extension/linalg/LinearTransformKeyPlanner.h"  // from @cyclops
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir::heir::cheddar {

#define GEN_PASS_DEF_CHEDDARPLANEVALUATIONKEYS
#include "lib/Dialect/Cheddar/Transforms/PlanEvaluationKeys.h.inc"

namespace {

func::FuncOp findClientSetup(ModuleOp module) {
  func::FuncOp found;
  module.walk([&](func::FuncOp func) {
    if (!hasInterfaceRole(func, kClientSetupRole)) return WalkResult::advance();
    found = func;
    return WalkResult::interrupt();
  });
  return found;
}

struct PlanEvaluationKeysPass
    : impl::CheddarPlanEvaluationKeysBase<PlanEvaluationKeysPass> {
  using CheddarPlanEvaluationKeysBase::CheddarPlanEvaluationKeysBase;

  void runOnOperation() override try {
    auto module = cast<ModuleOp>(getOperation());
    func::FuncOp setup = findClientSetup(module);
    if (!setup || !setup->hasAttr(kRotationKeysAttrName)) return;

    MakeParameterOp parameterOp;
    setup.walk([&](MakeParameterOp op) {
      parameterOp = op;
      return WalkResult::interrupt();
    });
    if (!parameterOp) {
      setup.emitOpError(
          "has no cheddar.make_parameter, so its scheme "
          "parameters are unknown");
      return signalPassFailure();
    }

    ArrayRef<int64_t> mainPrimes = parameterOp.getMainPrimes();
    ArrayRef<int64_t> auxPrimes = parameterOp.getAuxPrimes();
    int64_t logScale = parameterOp.getLogScale().getInt();
    if (logScale < 0 || logScale >= 64) {
      parameterOp.emitOpError("log_scale must be between 0 and 63");
      return signalPassFailure();
    }
    std::vector<std::pair<int, int>> levels;
    for (size_t i = 0; i < mainPrimes.size(); ++i)
      levels.emplace_back(i + 1, 0);
    int defaultLevel =
        parameterOp.getDefaultEncryptionLevel()
            ? parameterOp.getDefaultEncryptionLevelAttr().getInt()
            : static_cast<int>(mainPrimes.size()) - 1;
    // Match the Parameter constructed in the generated client.
    cyclops::Parameter<uint64_t> params(
        parameterOp.getLogN().getInt(), double(uint64_t{1} << logScale),
        defaultLevel, levels,
        std::vector<uint64_t>(mainPrimes.begin(), mainPrimes.end()),
        std::vector<uint64_t>(auxPrimes.begin(), auxPrimes.end()));
    if (auto weight = parameterOp.getDenseHammingWeightAttr())
      params.SetDenseHammingWeight(weight.getInt());
    if (auto weight = parameterOp.getSparseHammingWeightAttr())
      params.SetSparseHammingWeight(weight.getInt());

    cyclops::EvkRequest request;
    auto rotationKeys =
        setup->getAttrOfType<DenseI64ArrayAttr>(kRotationKeysAttrName);
    if (!rotationKeys || rotationKeys.size() % 2 != 0) {
      setup.emitOpError() << kRotationKeysAttrName
                          << " must be a dense i64 array of (distance, level) "
                             "pairs";
      return signalPassFailure();
    }
    ArrayRef<int64_t> pairs = rotationKeys.asArrayRef();
    for (size_t i = 0; i + 1 < pairs.size(); i += 2)
      request.AddRequest(pairs[i], pairs[i + 1]);

    if (auto shapes =
            setup->getAttrOfType<ArrayAttr>(kLinearTransformKeysAttrName)) {
      for (auto [index, attr] : llvm::enumerate(shapes)) {
        auto shape = dyn_cast<DictionaryAttr>(attr);
        auto indices =
            shape ? shape.getAs<DenseI32ArrayAttr>("indices") : nullptr;
        auto width = shape ? shape.getAs<IntegerAttr>("width") : nullptr;
        auto level = shape ? shape.getAs<IntegerAttr>("level") : nullptr;
        auto bs = shape ? shape.getAs<IntegerAttr>("bs") : nullptr;
        auto gs = shape ? shape.getAs<IntegerAttr>("gs") : nullptr;
        if (!indices || !width || !level || !bs || !gs) {
          setup.emitOpError()
              << kLinearTransformKeysAttrName << " entry " << index
              << " is missing a field; run cheddar-configure-crypto-context "
                 "first";
          return signalPassFailure();
        }
        auto diagonals = indices.asArrayRef();
        cyclops::AddLinearTransformRequiredKeys(
            request, params, width.getInt(),
            std::span<const int>(diagonals.data(), diagonals.size()),
            level.getInt(), bs.getInt(), gs.getInt());
      }
    }

    if (auto slots =
            setup->getAttrOfType<IntegerAttr>(kBootstrapSlotsAttrName)) {
      auto cts = setup->getAttrOfType<IntegerAttr>(kBootstrapNumCtsAttrName);
      auto stc = setup->getAttrOfType<IntegerAttr>(kBootstrapNumStcAttrName);
      auto ratio =
          setup->getAttrOfType<IntegerAttr>(kBootstrapLogMessageRatioAttrName);
      if (!cts || !stc || !ratio) {
        setup.emitOpError(
            "bootstrap key planning needs num_cts, num_stc and "
            "log_message_ratio");
        return signalPassFailure();
      }
      // The emitter hard-codes the imaginary-removing variant.
      cyclops::BootParameter bootstrap(params.max_level_, cts.getInt(),
                                       stc.getInt(), ratio.getInt());
      cyclops::AddBootstrapRequiredRotations(
          request, params, bootstrap, slots.getInt(),
          cyclops::BootVariant::kImaginaryRemoving);
    }

    SmallVector<int64_t> flattened;
    auto append = [&](int family, int rotation, const auto& key) {
      flattened.append({family, rotation, key.level,
                        static_cast<int64_t>(key.key_mode),
                        key.required_num_aux});
    };
    for (const auto& [key, count] : request.AllRequests())
      append(0, key.rot_idx, key);
    for (const auto& [key, count] : request.ConjugationRequests())
      append(1, 0, key);
    for (const auto& [key, count] : request.MultiplicationRequests())
      append(2, 0, key);
    for (const auto& [key, count] : request.RotatedMultiplicationRequests())
      append(3, key.rot_idx, key);

    OpBuilder builder(module.getContext());
    setup->setAttr(kEvaluationKeysAttrName,
                   builder.getDenseI64ArrayAttr(flattened));
    for (StringRef name :
         {kRotationKeysAttrName, kLinearTransformKeysAttrName,
          kBootstrapSlotsAttrName, kBootstrapNumCtsAttrName,
          kBootstrapNumStcAttrName, kBootstrapLogMessageRatioAttrName})
      setup->removeAttr(name);
  } catch (const std::exception& error) {
    getOperation()->emitError()
        << "Cyclops key planning failed: " << error.what();
    signalPassFailure();
  }
};

}  // namespace
}  // namespace mlir::heir::cheddar
