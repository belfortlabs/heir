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
#include "extension/boot/Mod1Parameters.h"               // from @cyclops
#include "extension/linalg/LinearTransformKeyPlanner.h"  // from @cyclops
#include "lib/Dialect/Cheddar/IR/CheddarAttributes.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/StringSwitch.h"         // from @llvm-project
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

// The x mod 1 approximation the attribute describes, on top of Cyclops'
// default for the same message ratio: the same resolution the emitted
// BootParameter performs, so the planned keys match the runtime's circuit.
std::optional<::cyclops::Mod1ParametersLiteral> toMod1Literal(
    EvalModAttr attr, int logMessageRatio) {
  if (!attr) return std::nullopt;
  ::cyclops::Mod1ParametersLiteral literal =
      ::cyclops::BootParameter::DefaultMod1(logMessageRatio);
  if (StringAttr type = attr.getType()) {
    literal.type = llvm::StringSwitch<::cyclops::Mod1Type>(type.getValue())
                       .Case("cos_hk", ::cyclops::Mod1Type::kCosHK)
                       .Case("cos_hk_even", ::cyclops::Mod1Type::kCosHKEven)
                       .Case("cos_cheby", ::cyclops::Mod1Type::kCosCheby)
                       .Case("sin_cheby", ::cyclops::Mod1Type::kSinCheby)
                       .Case("exp_complex", ::cyclops::Mod1Type::kExpComplex)
                       .Default(literal.type);
  }
  if (auto degree = attr.getDegree()) literal.degree = *degree;
  if (auto interval = attr.getInterval()) literal.interval = *interval;
  if (auto reduction = attr.getLogIntervalReduction())
    literal.log_interval_reduction = *reduction;
  if (auto invDegree = attr.getInvDegree()) literal.inv_degree = *invDegree;
  if (StringAttr invType = attr.getInvType())
    literal.inv_type = invType.getValue() == "cheby"
                           ? ::cyclops::Mod1InvType::kArcsineCheby
                           : ::cyclops::Mod1InvType::kArcsineTaylor;
  if (FloatAttr invInterval = attr.getInvInterval())
    literal.inv_interval = invInterval.getValueAsDouble();
  return literal;
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
                     DenseI64ArrayAttr rotationKeys,
                     DenseI64ArrayAttr multiplicationKeys, ArrayAttr shapes,
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
        std::pair<int, int>(baseMain, baseTerminal),
        static_cast<int>(parameterSet.getDefaultNumAux().value_or(-1)));
    // The dense weight goes first: the sparse one must stay below it.
    if (auto weight = parameterSet.getDenseHammingWeight())
      params.SetDenseHammingWeight(*weight);
    if (auto weight = parameterSet.getSparseHammingWeight())
      params.SetSparseHammingWeight(*weight);
    if (FloatAttr budget = parameterSet.getMaxLogPq())
      params.SetMaxLogPQ(budget.getValueAsDouble());
    if (BoolAttr levelSpecific = parameterSet.getLevelSpecificKs())
      params.SetLevelSpecificKS(levelSpecific.getValue());
    if (auto cap = parameterSet.getMaxKeySwitchAux())
      params.SetMaxKeySwitchAux(*cap);

    ::cyclops::EvkRequest request;
    ArrayRef<int64_t> pairs = rotationKeys.asArrayRef();
    for (size_t i = 0; i + 1 < pairs.size(); i += 2)
      request.AddRequest(pairs[i], pairs[i + 1]);
    // Default-preferred: the client satisfies these with the default
    // multiplication key where the ring's budget holds one, and builds a key
    // for the level otherwise.
    if (multiplicationKeys)
      for (int64_t level : multiplicationKeys.asArrayRef())
        request.RequestMultiplicationKey(level, ::cyclops::KeyMode::kDefault);

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
          bootstrapConfig.getNumStcLevels(), ratio,
          toMod1Literal(bootstrapConfig.getEvalMod(), ratio));
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
        kRotationKeysAttrName, kMultiplicationKeysAttrName,
        kLinearTransformKeysAttrName, kBootstrapSlotsAttrName,
        kBootstrapConfigAttrName};
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
    auto multiplicationKeys =
        setup->getAttrOfType<DenseI64ArrayAttr>(kMultiplicationKeysAttrName);
    if (setup->hasAttr(kMultiplicationKeysAttrName) && !multiplicationKeys) {
      setup.emitOpError() << kMultiplicationKeysAttrName
                          << " must be a dense i64 array of levels";
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
            ? plan<uint32_t>(setup, parameterSet, rotationKeys,
                             multiplicationKeys, shapes, bootstrapSlots,
                             bootstrapConfig)
            : plan<uint64_t>(setup, parameterSet, rotationKeys,
                             multiplicationKeys, shapes, bootstrapSlots,
                             bootstrapConfig);
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
