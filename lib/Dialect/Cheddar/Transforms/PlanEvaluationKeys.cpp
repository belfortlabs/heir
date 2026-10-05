#include "lib/Dialect/Cheddar/Transforms/PlanEvaluationKeys.h"

#include <cstdint>
#include <memory>

#include "cyclops_planner.h"  // from @cyclops
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

  void runOnOperation() override {
    auto module = cast<ModuleOp>(getOperation());
    func::FuncOp setup = findClientSetup(module);
    if (!setup) return;
    const StringRef planningAttrs[] = {
        kRotationKeysAttrName,    kLinearTransformKeysAttrName,
        kBootstrapSlotsAttrName,  kBootstrapNumCtsAttrName,
        kBootstrapNumStcAttrName, kBootstrapLogMessageRatioAttrName};
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
    SmallVector<int32_t> levels;
    for (size_t i = 0; i < mainPrimes.size(); ++i)
      levels.append({static_cast<int32_t>(i + 1), 0});
    int defaultLevel =
        parameterOp.getDefaultEncryptionLevel()
            ? parameterOp.getDefaultEncryptionLevelAttr().getInt()
            : static_cast<int>(mainPrimes.size()) - 1;

    char* error = nullptr;
    // Reports a failed planner call; NULL error means allocation failed.
    auto failed = [&](bool ok) {
      if (ok) return false;
      getOperation()->emitError() << "Cyclops key planning failed: "
                                  << (error ? error : "out of memory");
      cyclops_free_error(error);
      signalPassFailure();
      return true;
    };

    // Match the Parameter constructed in the generated client.
    std::unique_ptr<cyclops_params, decltype(&cyclops_params_free)> params(
        cyclops_params_create(
            parameterOp.getLogN().getInt(), double(uint64_t{1} << logScale),
            defaultLevel, levels.data(), mainPrimes.size(),
            reinterpret_cast<const uint64_t*>(mainPrimes.data()),
            mainPrimes.size(),
            reinterpret_cast<const uint64_t*>(auxPrimes.data()),
            auxPrimes.size(), nullptr, 0, nullptr, -1, CYCLOPS_RING_STANDARD,
            &error),
        cyclops_params_free);
    if (failed(params != nullptr)) return;
    if (auto weight = parameterOp.getDenseHammingWeightAttr())
      if (failed(cyclops_params_set_dense_hamming_weight(
                     params.get(), weight.getInt(), &error) == 0))
        return;
    if (auto weight = parameterOp.getSparseHammingWeightAttr())
      if (failed(cyclops_params_set_sparse_hamming_weight(
                     params.get(), weight.getInt(), &error) == 0))
        return;

    std::unique_ptr<cyclops_evk_request, decltype(&cyclops_evk_request_free)>
        request(cyclops_evk_request_create(), cyclops_evk_request_free);
    if (failed(request != nullptr)) return;
    ArrayRef<int64_t> pairs = rotationKeys.asArrayRef();
    for (size_t i = 0; i + 1 < pairs.size(); i += 2)
      if (failed(cyclops_evk_request_add_request(
                     request.get(), pairs[i], pairs[i + 1],
                     CYCLOPS_KEY_MODE_INHERIT, -1, &error) == 0))
        return;

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
        if (failed(cyclops_add_linear_transform_required_keys(
                       request.get(), params.get(), width.getInt(),
                       diagonals.data(), diagonals.size(), level.getInt(),
                       bs.getInt(), gs.getInt(), CYCLOPS_KEY_MODE_INHERIT,
                       &error) == 0))
          return;
      }
    }

    if (auto slots =
            setup->getAttrOfType<IntegerAttr>(kBootstrapSlotsAttrName)) {
      auto cts = setup->getAttrOfType<IntegerAttr>(kBootstrapNumCtsAttrName);
      auto stc = setup->getAttrOfType<IntegerAttr>(kBootstrapNumStcAttrName);
      auto ratio =
          setup->getAttrOfType<IntegerAttr>(kBootstrapLogMessageRatioAttrName);
      // The emitter hard-codes the imaginary-removing variant.
      if (failed(cyclops_add_bootstrap_required_rotations(
                     request.get(), params.get(), cts.getInt(), stc.getInt(),
                     ratio.getInt(), nullptr, slots.getInt(),
                     CYCLOPS_BOOT_VARIANT_IMAGINARY_REMOVING,
                     CYCLOPS_KEY_MODE_INHERIT, &error) == 0))
        return;
    }

    SmallVector<cyclops_key> keys(
        cyclops_evk_request_keys(request.get(), nullptr, 0));
    cyclops_evk_request_keys(request.get(), keys.data(), keys.size());
    SmallVector<int64_t> flattened;
    for (const cyclops_key& key : keys)
      flattened.append({key.family, key.rotation, key.level, key.key_mode,
                        key.required_num_aux});

    OpBuilder builder(module.getContext());
    setup->setAttr(kEvaluationKeysAttrName,
                   builder.getDenseI64ArrayAttr(flattened));
    for (StringRef name : planningAttrs) setup->removeAttr(name);
  }
};

}  // namespace
}  // namespace mlir::heir::cheddar
