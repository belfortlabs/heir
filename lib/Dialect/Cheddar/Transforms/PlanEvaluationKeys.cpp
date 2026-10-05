#include "lib/Dialect/Cheddar/Transforms/PlanEvaluationKeys.h"

#include <cstdint>
#include <exception>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "core/EvkRequest.h"                             // from @cyclops
#include "core/Parameter.h"                              // from @cyclops
#include "extension/boot/BootKeyPlanner.h"               // from @cyclops
#include "extension/boot/BootParameter.h"                // from @cyclops
#include "extension/linalg/LinearTransformKeyPlanner.h"  // from @cyclops
#include "extension/max/MaxPoolPlanner.h"                // from @cyclops
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Target/Cheddar/MaxPoolCyclopsConfig.h"
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

// Disables Cyclops' small-window league level cap, which HEIR's level model
// leaves out (requirement R2 in cyclops_maxpool_requirements.md), when the
// Cyclops planner has an option for it.
template <typename Config>
void disableLeagueLevelCap(Config& config) {
  if constexpr (requires { config.cap_league_level; })
    config.cap_league_level = false;
}

struct PlanEvaluationKeysPass
    : impl::CheddarPlanEvaluationKeysBase<PlanEvaluationKeysPass> {
  using CheddarPlanEvaluationKeysBase::CheddarPlanEvaluationKeysBase;

  void runOnOperation() override try {
    auto module = cast<ModuleOp>(getOperation());
    func::FuncOp setup = findClientSetup(module);
    if (!setup) return;
    const StringRef planningAttrs[] = {
        kRotationKeysAttrName,    kLinearTransformKeysAttrName,
        kBootstrapSlotsAttrName,  kBootstrapNumCtsAttrName,
        kBootstrapNumStcAttrName, kBootstrapLogMessageRatioAttrName,
        kMaxPoolKeysAttrName};
    if (llvm::none_of(planningAttrs,
                      [&](StringRef name) { return setup->hasAttr(name); }))
      return;

    auto rotationKeys =
        setup->getAttrOfType<DenseI64ArrayAttr>(kRotationKeysAttrName);
    if (!rotationKeys || rotationKeys.size() % 2 != 0) {
      setup.emitOpError()
          << kRotationKeysAttrName
          << " must be a dense i64 array of (distance, level) pairs";
      return signalPassFailure();
    }
    auto shapes = setup->getAttrOfType<ArrayAttr>(kLinearTransformKeysAttrName);
    if (setup->hasAttr(kLinearTransformKeysAttrName) && !shapes) {
      setup.emitOpError() << kLinearTransformKeysAttrName
                          << " must be an array";
      return signalPassFailure();
    }
    const StringRef bootstrapAttrs[] = {
        kBootstrapSlotsAttrName, kBootstrapNumCtsAttrName,
        kBootstrapNumStcAttrName, kBootstrapLogMessageRatioAttrName};
    if (llvm::any_of(bootstrapAttrs,
                     [&](StringRef name) { return setup->hasAttr(name); })) {
      for (StringRef name : bootstrapAttrs) {
        auto value = setup->getAttrOfType<IntegerAttr>(name);
        if (!value || !value.getType().isInteger(64)) {
          setup.emitOpError()
              << name << " must be i64 when planning bootstrap keys";
          return signalPassFailure();
        }
      }
    }

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
    if (auxPrimes.empty()) {
      parameterOp.emitOpError(
          "requires at least one auxiliary prime for Cyclops key planning");
      return signalPassFailure();
    }
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
    ArrayRef<int64_t> pairs = rotationKeys.asArrayRef();
    for (size_t i = 0; i + 1 < pairs.size(); i += 2)
      request.AddRequest(pairs[i], pairs[i + 1]);

    if (shapes) {
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
        for (StringRef field : {"width", "level", "bs", "gs"}) {
          if (!shape.getAs<IntegerAttr>(field).getType().isInteger(64)) {
            setup.emitOpError()
                << kLinearTransformKeysAttrName << " entry " << index
                << " requires an i64 " << field << " field";
            return signalPassFailure();
          }
        }
        auto diagonals = indices.asArrayRef();
        cyclops::AddLinearTransformRequiredKeys(
            request, params, width.getInt(),
            std::span<const int>(diagonals.data(), diagonals.size()),
            level.getInt(), bs.getInt(), gs.getInt());
      }
    }

    // The bootstrap parameters, shared by the bootstraps and the max pools.
    std::optional<cyclops::BootParameter> bootstrap;
    if (auto slots =
            setup->getAttrOfType<IntegerAttr>(kBootstrapSlotsAttrName)) {
      auto cts = setup->getAttrOfType<IntegerAttr>(kBootstrapNumCtsAttrName);
      auto stc = setup->getAttrOfType<IntegerAttr>(kBootstrapNumStcAttrName);
      auto ratio =
          setup->getAttrOfType<IntegerAttr>(kBootstrapLogMessageRatioAttrName);
      bootstrap.emplace(params.max_level_, cts.getInt(), stc.getInt(),
                        ratio.getInt());
      // The emitter hard-codes the imaginary-removing variant.
      cyclops::AddBootstrapRequiredRotations(
          request, params, *bootstrap, slots.getInt(),
          cyclops::BootVariant::kImaginaryRemoving);
    }

    if (auto maxPools = setup->getAttrOfType<ArrayAttr>(kMaxPoolKeysAttrName);
        maxPools && !maxPools.empty()) {
      if (!bootstrap) {
        setup.emitOpError() << kMaxPoolKeysAttrName
                            << " requires the bootstrap planning attributes";
        return signalPassFailure();
      }
      for (auto [index, attr] : llvm::enumerate(maxPools)) {
        auto shape = dyn_cast<DictionaryAttr>(attr);
        auto field = [&](StringRef name) -> IntegerAttr {
          return shape ? shape.getAs<IntegerAttr>(name) : nullptr;
        };
        auto ceilMode = shape ? shape.getAs<BoolAttr>("ceil_mode") : nullptr;
        auto valueBound =
            shape ? shape.getAs<FloatAttr>("value_bound") : nullptr;
        if (!field("num_slots") || !field("input_length") ||
            !field("window_size") || !field("stride") || !field("dilation") ||
            !field("level") || !field("level_consumption") || !ceilMode ||
            !valueBound) {
          setup.emitOpError()
              << kMaxPoolKeysAttrName << " entry " << index
              << " is missing a field; run cheddar-configure-crypto-context "
                 "first";
          return signalPassFailure();
        }
        std::optional<cyclops::MaxPoolConfig> config = toCyclopsMaxPoolConfig(
            {field("num_slots").getInt(), field("input_length").getInt(),
             field("window_size").getInt(), field("stride").getInt(),
             field("dilation").getInt(), ceilMode.getValue(),
             valueBound.getValueAsDouble()});
        if (!config) {
          setup.emitOpError() << kMaxPoolKeysAttrName << " entry " << index
                              << " does not fit a Cyclops MaxPoolConfig";
          return signalPassFailure();
        }
        disableLeagueLevelCap(*config);
        // The generated code checks the same levels at run time, against the
        // compiled MaxPool. Without the cap option, the small-window league
        // level cap lowers the levels of a padded window of 2 to 7 slots when
        // its input level is more than the selector depth + 1 above the
        // gather stages.
        int level = field("level").getInt();
        int levelConsumption = field("level_consumption").getInt();
        cyclops::MaxPoolPlan plan =
            cyclops::PlanMaxPool(*config, level, bootstrap->GetEndLevel());
        if (level - plan.output_level != levelConsumption) {
          setup.emitOpError()
              << "cannot plan the evaluation keys of max pool " << index
              << ": at input level " << level << " and bootstrap end level "
              << bootstrap->GetEndLevel() << ", Cyclops' MaxPool drops "
              << level - plan.output_level << " levels, but HEIR planned "
              << levelConsumption
              << "; Cyclops' small-window league level cap causes this when "
                 "MaxPoolConfig has no cap_league_level option";
          return signalPassFailure();
        }
        // The emitter hard-codes the imaginary-removing variant.
        cyclops::AddMaxPoolRequiredKeys(
            request, params, *bootstrap, *config, level,
            cyclops::BootVariant::kImaginaryRemoving);
      }
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
    for (StringRef name : planningAttrs) setup->removeAttr(name);
  } catch (const std::exception& error) {
    getOperation()->emitError()
        << "Cyclops key planning failed: " << error.what();
    signalPassFailure();
  }
};

}  // namespace
}  // namespace mlir::heir::cheddar
