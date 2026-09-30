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
#include "lib/Dialect/Cheddar/IR/CheddarAttributes.h"
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

template <typename Word>
std::vector<Word> primesOf(DenseI64ArrayAttr primes) {
  std::vector<Word> result;
  if (!primes) return result;
  for (int64_t prime : primes.asArrayRef())
    result.push_back(static_cast<Word>(prime));
  return result;
}

struct PlanEvaluationKeysPass
    : impl::CheddarPlanEvaluationKeysBase<PlanEvaluationKeysPass> {
  using CheddarPlanEvaluationKeysBase::CheddarPlanEvaluationKeysBase;

  // Resolves the recorded key requirements against the runtime's own planner,
  // with a Parameter built exactly as the emitted client constructs it.
  template <typename Word>
  LogicalResult plan(func::FuncOp setup, ParameterSetAttr parameterSet,
                     DenseI64ArrayAttr rotationKeys, ArrayAttr shapes,
                     IntegerAttr bootstrapSlots,
                     BootstrapConfigAttr bootstrapConfig) {
    std::vector<std::pair<int, int>> levels;
    for (auto [numMain, numTerminal] : parameterSet.getLevelPairs())
      levels.emplace_back(numMain, numTerminal);
    auto [baseMain, baseTerminal] = parameterSet.getAdditionalBasePair();
    ::cyclops::Parameter<Word> params(
        parameterSet.getLogN(),
        static_cast<double>(uint64_t{1} << parameterSet.getLogScale()),
        parameterSet.getDefaultEncryptionLevelOrDefault(), levels,
        primesOf<Word>(parameterSet.getMainPrimes()),
        primesOf<Word>(parameterSet.getAuxPrimes()),
        primesOf<Word>(parameterSet.getTerminalPrimes()),
        std::pair<int, int>(baseMain, baseTerminal));
    // The dense weight goes first: the sparse one must stay below it.
    if (auto weight = parameterSet.getDenseHammingWeight())
      params.SetDenseHammingWeight(*weight);
    if (auto weight = parameterSet.getSparseHammingWeight())
      params.SetSparseHammingWeight(*weight);

    ::cyclops::EvkRequest request;
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
          return setup.emitOpError()
                 << kLinearTransformKeysAttrName << " entry " << index
                 << " is missing a field; run cheddar-configure-crypto-context "
                    "first";
        }
        for (StringRef field : {"width", "level", "bs", "gs"}) {
          if (!shape.getAs<IntegerAttr>(field).getType().isInteger(64)) {
            return setup.emitOpError()
                   << kLinearTransformKeysAttrName << " entry " << index
                   << " requires an i64 " << field << " field";
          }
        }
        auto diagonals = indices.asArrayRef();
        ::cyclops::AddLinearTransformRequiredKeys(
            request, params, width.getInt(),
            std::span<const int>(diagonals.data(), diagonals.size()),
            level.getInt(), bs.getInt(), gs.getInt());
      }
    }

    if (bootstrapSlots) {
      int ratio = bootstrapConfig.getLogMessageRatio().value_or(
          ::cyclops::BootParameter::kDefaultLogMessageRatio);
      // The emitter hard-codes the imaginary-removing variant.
      ::cyclops::BootParameter bootstrap(
          params.max_level_, bootstrapConfig.getNumCtsLevels(),
          bootstrapConfig.getNumStcLevels(), ratio);
      if (bootstrap.GetNumEvalModLevels() !=
          bootstrapConfig.getNumEvalModLevels()) {
        return setup.emitOpError()
               << "the EvalMod approximation consumes "
               << bootstrap.GetNumEvalModLevels()
               << " levels, but the bootstrap config reserves "
               << bootstrapConfig.getNumEvalModLevels();
      }
      ::cyclops::AddBootstrapRequiredRotations(
          request, params, bootstrap, bootstrapSlots.getInt(),
          ::cyclops::BootVariant::kImaginaryRemoving);
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

    OpBuilder builder(setup.getContext());
    setup->setAttr(kEvaluationKeysAttrName,
                   builder.getDenseI64ArrayAttr(flattened));
    return success();
  }

  void runOnOperation() override try {
    auto module = cast<ModuleOp>(getOperation());
    func::FuncOp setup = findClientSetup(module);
    if (!setup) return;
    const StringRef planningAttrs[] = {
        kRotationKeysAttrName, kLinearTransformKeysAttrName,
        kBootstrapSlotsAttrName, kBootstrapConfigAttrName};
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
    auto bootstrapSlots =
        setup->getAttrOfType<IntegerAttr>(kBootstrapSlotsAttrName);
    auto bootstrapConfig =
        setup->getAttrOfType<BootstrapConfigAttr>(kBootstrapConfigAttrName);
    if (setup->hasAttr(kBootstrapSlotsAttrName) ||
        setup->hasAttr(kBootstrapConfigAttrName)) {
      if (!bootstrapSlots || !bootstrapSlots.getType().isInteger(64)) {
        setup.emitOpError() << kBootstrapSlotsAttrName
                            << " must be i64 when planning bootstrap keys";
        return signalPassFailure();
      }
      if (!bootstrapConfig) {
        setup.emitOpError() << kBootstrapConfigAttrName
                            << " must be a #cheddar.bootstrap_config when "
                               "planning bootstrap keys";
        return signalPassFailure();
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
    ParameterSetAttr parameterSet = parameterOp.getParameterSet();
    if (parameterSet.getAuxPrimes().empty()) {
      parameterOp.emitOpError(
          "requires at least one auxiliary prime for Cyclops key planning");
      return signalPassFailure();
    }

    LogicalResult planned =
        parameterSet.getWordBitsOrDefault() == 32
            ? plan<uint32_t>(setup, parameterSet, rotationKeys, shapes,
                             bootstrapSlots, bootstrapConfig)
            : plan<uint64_t>(setup, parameterSet, rotationKeys, shapes,
                             bootstrapSlots, bootstrapConfig);
    if (failed(planned)) return signalPassFailure();
    for (StringRef name : planningAttrs) setup->removeAttr(name);
  } catch (const std::exception& error) {
    getOperation()->emitError()
        << "Cyclops key planning failed: " << error.what();
    signalPassFailure();
  }
};

}  // namespace
}  // namespace mlir::heir::cheddar
