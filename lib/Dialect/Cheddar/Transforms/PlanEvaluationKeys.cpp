#include "lib/Dialect/Cheddar/Transforms/PlanEvaluationKeys.h"

#include <cstdint>
#include <memory>
#include <optional>

#include "cyclops_planner.h"  // from @cyclops
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
// default: the same resolution the emitted BootParameter performs, so the
// planned keys match the runtime's circuit.
std::optional<cyclops_mod1> toMod1(EvalModAttr attr) {
  if (!attr) return std::nullopt;
  cyclops_mod1 mod1 = cyclops_default_mod1();
  if (StringAttr type = attr.getType()) {
    mod1.type = llvm::StringSwitch<int>(type.getValue())
                    .Case("cos_hk", CYCLOPS_MOD1_COS_HK)
                    .Case("cos_hk_even", CYCLOPS_MOD1_COS_HK_EVEN)
                    .Case("cos_cheby", CYCLOPS_MOD1_COS_CHEBY)
                    .Case("sin_cheby", CYCLOPS_MOD1_SIN_CHEBY)
                    .Case("exp_complex", CYCLOPS_MOD1_EXP_COMPLEX)
                    .Default(mod1.type);
  }
  if (auto degree = attr.getDegree()) mod1.degree = *degree;
  if (auto interval = attr.getInterval()) mod1.interval = *interval;
  if (auto reduction = attr.getLogIntervalReduction())
    mod1.log_interval_reduction = *reduction;
  if (auto invDegree = attr.getInvDegree()) mod1.inv_degree = *invDegree;
  if (StringAttr invType = attr.getInvType())
    mod1.inv_type = invType.getValue() == "cheby"
                        ? CYCLOPS_MOD1_INV_ARCSINE_CHEBY
                        : CYCLOPS_MOD1_INV_ARCSINE_TAYLOR;
  if (FloatAttr invInterval = attr.getInvInterval())
    mod1.inv_interval = invInterval.getValueAsDouble();
  return mod1;
}

// Cyclops' BootParameter::kDefaultLogMessageRatio, which the emitted client
// falls back to when the bootstrap config gives no ratio.
constexpr int kDefaultLogMessageRatio = 5;

const uint64_t* primesOf(DenseI64ArrayAttr primes) {
  return primes ? reinterpret_cast<const uint64_t*>(primes.asArrayRef().data())
                : nullptr;
}

size_t sizeOf(DenseI64ArrayAttr primes) { return primes ? primes.size() : 0; }

struct PlanEvaluationKeysPass
    : impl::CheddarPlanEvaluationKeysBase<PlanEvaluationKeysPass> {
  using CheddarPlanEvaluationKeysBase::CheddarPlanEvaluationKeysBase;

  // Resolves the recorded key requirements against the runtime's own planner,
  // with a Parameter built exactly as the emitted client constructs it.
  LogicalResult plan(func::FuncOp setup, ParameterSetAttr parameterSet,
                     DenseI64ArrayAttr rotationKeys,
                     DenseI64ArrayAttr multiplicationKeys, ArrayAttr shapes,
                     IntegerAttr bootstrapSlots,
                     BootstrapConfigAttr bootstrapConfig) {
    char* error = nullptr;
    // Reports a failed planner call; NULL error means allocation failed.
    auto failed = [&](bool ok) {
      if (ok) return false;
      setup.emitOpError() << "Cyclops key planning failed: "
                          << (error ? error : "out of memory");
      cyclops_free_error(error);
      return true;
    };

    SmallVector<int32_t> levels;
    for (auto [numMain, numTerminal] : parameterSet.getLevelPairs())
      levels.append(
          {static_cast<int32_t>(numMain), static_cast<int32_t>(numTerminal)});
    auto [baseMain, baseTerminal] = parameterSet.getAdditionalBasePair();
    const int32_t additionalBase[2] = {static_cast<int32_t>(baseMain),
                                       static_cast<int32_t>(baseTerminal)};
    std::unique_ptr<cyclops_params, decltype(&cyclops_params_free)> params(
        cyclops_params_create(
            parameterSet.getLogN(),
            static_cast<double>(uint64_t{1} << parameterSet.getLogScale()),
            parameterSet.getDefaultEncryptionLevelOrDefault(), levels.data(),
            levels.size() / 2, primesOf(parameterSet.getMainPrimes()),
            sizeOf(parameterSet.getMainPrimes()),
            primesOf(parameterSet.getAuxPrimes()),
            sizeOf(parameterSet.getAuxPrimes()),
            primesOf(parameterSet.getTerminalPrimes()),
            sizeOf(parameterSet.getTerminalPrimes()), additionalBase,
            static_cast<int>(parameterSet.getDefaultNumAux().value_or(-1)),
            CYCLOPS_RING_STANDARD,
            static_cast<int>(parameterSet.getWordBitsOrDefault()), &error),
        cyclops_params_free);
    if (failed(params != nullptr)) return failure();
    // The dense weight goes first: the sparse one must stay below it.
    if (auto weight = parameterSet.getDenseHammingWeight())
      if (failed(cyclops_params_set_dense_hamming_weight(params.get(), *weight,
                                                         &error) == 0))
        return failure();
    if (auto weight = parameterSet.getSparseHammingWeight())
      if (failed(cyclops_params_set_sparse_hamming_weight(params.get(), *weight,
                                                          &error) == 0))
        return failure();
    if (FloatAttr budget = parameterSet.getMaxLogPq())
      if (failed(cyclops_params_set_max_log_pq(
                     params.get(), budget.getValueAsDouble(), &error) == 0))
        return failure();
    if (BoolAttr levelSpecific = parameterSet.getLevelSpecificKs())
      if (failed(cyclops_params_set_level_specific_ks(
                     params.get(), levelSpecific.getValue(), &error) == 0))
        return failure();
    if (auto cap = parameterSet.getMaxKeySwitchAux())
      if (failed(cyclops_params_set_max_key_switch_aux(params.get(), *cap,
                                                       &error) == 0))
        return failure();

    std::unique_ptr<cyclops_evk_request, decltype(&cyclops_evk_request_free)>
        request(cyclops_evk_request_create(), cyclops_evk_request_free);
    if (failed(request != nullptr)) return failure();
    ArrayRef<int64_t> pairs = rotationKeys.asArrayRef();
    for (size_t i = 0; i + 1 < pairs.size(); i += 2)
      if (failed(cyclops_evk_request_add_request(
                     request.get(), pairs[i], pairs[i + 1],
                     CYCLOPS_KEY_MODE_INHERIT, -1, &error) == 0))
        return failure();
    // Default-preferred: the client satisfies these with the default
    // multiplication key where the ring's budget holds one, and builds a key
    // for the level otherwise.
    if (multiplicationKeys)
      for (int64_t level : multiplicationKeys.asArrayRef())
        if (failed(cyclops_evk_request_request_multiplication_key(
                       request.get(), level, CYCLOPS_KEY_MODE_DEFAULT, -1,
                       &error) == 0))
          return failure();

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
        if (failed(cyclops_add_linear_transform_required_keys(
                       request.get(), params.get(), width.getInt(),
                       diagonals.data(), diagonals.size(), level.getInt(),
                       bs.getInt(), gs.getInt(), CYCLOPS_KEY_MODE_INHERIT,
                       &error) == 0))
          return failure();
      }
    }

    if (bootstrapSlots) {
      int ratio = bootstrapConfig.getLogMessageRatio().value_or(
          kDefaultLogMessageRatio);
      std::optional<cyclops_mod1> mod1 = toMod1(bootstrapConfig.getEvalMod());
      int evalModLevels = 0;
      if (failed(cyclops_mod1_depth(mod1 ? &*mod1 : nullptr, ratio,
                                    &evalModLevels, &error) == 0))
        return failure();
      if (evalModLevels != bootstrapConfig.getNumEvalModLevels()) {
        return setup.emitOpError()
               << "the EvalMod approximation consumes " << evalModLevels
               << " levels, but the bootstrap config reserves "
               << bootstrapConfig.getNumEvalModLevels();
      }
      // The emitter hard-codes the imaginary-removing variant.
      if (failed(cyclops_add_bootstrap_required_rotations(
                     request.get(), params.get(),
                     bootstrapConfig.getNumCtsLevels(),
                     bootstrapConfig.getNumStcLevels(), ratio,
                     mod1 ? &*mod1 : nullptr, bootstrapSlots.getInt(),
                     CYCLOPS_BOOT_VARIANT_IMAGINARY_REMOVING,
                     CYCLOPS_KEY_MODE_INHERIT, &error) == 0))
        return failure();
    }

    SmallVector<cyclops_key> keys(
        cyclops_evk_request_keys(request.get(), nullptr, 0));
    cyclops_evk_request_keys(request.get(), keys.data(), keys.size());
    SmallVector<int64_t> flattened;
    for (const cyclops_key& key : keys)
      flattened.append({key.family, key.rotation, key.level, key.key_mode,
                        key.required_num_aux});

    OpBuilder builder(setup.getContext());
    setup->setAttr(kEvaluationKeysAttrName,
                   builder.getDenseI64ArrayAttr(flattened));
    return success();
  }

  void runOnOperation() override {
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

    if (failed(plan(setup, parameterSet, rotationKeys, multiplicationKeys,
                    shapes, bootstrapSlots, bootstrapConfig)))
      return signalPassFailure();
    for (StringRef name : planningAttrs) setup->removeAttr(name);
  }
};

}  // namespace
}  // namespace mlir::heir::cheddar
