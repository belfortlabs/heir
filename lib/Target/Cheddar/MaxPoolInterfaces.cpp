#include "lib/Target/Cheddar/MaxPoolInterfaces.h"

#include <cstdint>
#include <optional>
#include <string>

#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Kernel/IR/KernelDialect.h"
#include "lib/Dialect/Kernel/IR/KernelOps.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Target/Cheddar/CyclopsPlanner.h"
#include "lib/Target/Cheddar/MaxPoolPlan.h"
#include "lib/Utils/MaxPoolUtils.h"
#include "llvm/include/llvm/ADT/Twine.h"                 // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {

namespace {

// The level model needs Cyclops' planner library, which heir-opt loads at run
// time. The interfaces cannot report a diagnostic, so a missing library stops
// heir-opt with a message instead of a wrong level plan.
void requirePlanner() {
  const std::string& error = getCyclopsPlanner().maxPoolError;
  if (!error.empty())
    llvm::report_fatal_error(
        llvm::Twine("the level plan of a max pool needs libcyclops_planner "
                    "with the MaxPool functions on the library path: ") +
            error,
        /*gen_crash_diag=*/false);
}

// The verifier of kernel.max_pool accepts only shapes that the planner
// accepts, so a rejection here is a disagreement between the two.
MaxPoolRequirements getRequirements(kernel::MaxPoolOp op) {
  requirePlanner();
  std::string error;
  std::optional<MaxPoolRequirements> requirements = getMaxPoolRequirements(
      {static_cast<int64_t>(op.getNumSlots()),
       static_cast<int64_t>(op.getInputLength()),
       static_cast<int64_t>(op.getWindowSize()),
       static_cast<int64_t>(op.getStride()),
       static_cast<int64_t>(op.getDilation()), op.getCeilMode(),
       op.getValueBound().convertToDouble()},
      &error);
  if (!requirements)
    llvm::report_fatal_error(
        llvm::Twine("the Cyclops max pool planner rejects a kernel.max_pool "
                    "that the verifier accepts: ") +
        error);
  return *requirements;
}

struct KernelMaxPoolReducesLevelModel
    : public ReducesLevelOpInterface::ExternalModel<
          KernelMaxPoolReducesLevelModel, kernel::MaxPoolOp> {
  int getLevelsToDrop(Operation* op) const {
    return getRequirements(cast<kernel::MaxPoolOp>(op)).levelDrop;
  }

  int getRequiredInputLevels(Operation* op) const {
    return getRequirements(cast<kernel::MaxPoolOp>(op)).minimumInputLevel;
  }

  SmallVector<OpOperand*> getOperandsToReduce(
      Operation* op, const DataFlowSolver* solver) const {
    return {&op->getOpOperand(0)};
  }
};

struct KernelMaxPoolBootstrapsInternallyModel
    : public BootstrapsInternallyOpInterface::ExternalModel<
          KernelMaxPoolBootstrapsInternallyModel, kernel::MaxPoolOp> {
  int getMinimumBootstrapEndLevel(Operation* op) const {
    return getRequirements(cast<kernel::MaxPoolOp>(op))
        .minimumBootstrapEndLevel;
  }
};

// A max pool lowers to kernel.max_pool, whose level cost depends on the slot
// count. The pool's result layout is compact, so it does not give the slot
// count: layout propagation records it on the module, and every ciphertext has
// that many slots.
struct LinalgMaxPoolReducesLevelModel
    : public ReducesLevelOpInterface::ExternalModel<
          LinalgMaxPoolReducesLevelModel, linalg::PoolingNcwMaxOp> {
  static std::optional<MaxPoolRequirements> getRequirements(Operation* op) {
    if (!op->hasAttr(tensor_ext::TensorExtDialect::kLayoutAttrName))
      return std::nullopt;
    std::string reason;
    FailureOr<SupportedMaxPool> pool =
        getSupportedMaxPool(cast<linalg::PoolingNcwMaxOp>(op), reason);
    int64_t numSlots = getLayoutSlotCount(op, /*minSlotCount=*/0);
    if (failed(pool) || numSlots <= 0) return std::nullopt;
    requirePlanner();
    std::string error;
    std::optional<MaxPoolRequirements> requirements =
        getMaxPoolRequirements(pool->getKernelShape(numSlots), &error);
    if (!requirements)
      llvm::report_fatal_error(
          llvm::Twine("the Cyclops max pool planner rejects the "
                      "kernel.max_pool of a linalg.pooling_ncw_max: ") +
              error,
          /*gen_crash_diag=*/false);
    return requirements;
  }

  // Before layout propagation, count one level, as for the other linalg ops.
  int getLevelsToDrop(Operation* op) const {
    std::optional<MaxPoolRequirements> requirements = getRequirements(op);
    return requirements ? requirements->levelDrop : 1;
  }

  int getRequiredInputLevels(Operation* op) const {
    std::optional<MaxPoolRequirements> requirements = getRequirements(op);
    return requirements ? requirements->minimumInputLevel : 1;
  }

  SmallVector<OpOperand*> getOperandsToReduce(
      Operation* op, const DataFlowSolver* solver) const {
    return {&op->getOpOperand(0)};
  }
};

}  // namespace

void registerMaxPoolInterfaceExternalModels(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, kernel::KernelDialect* dialect) {
    kernel::MaxPoolOp::attachInterface<KernelMaxPoolReducesLevelModel,
                                       KernelMaxPoolBootstrapsInternallyModel>(
        *ctx);
  });
  registry.addExtension(+[](MLIRContext* ctx, linalg::LinalgDialect* dialect) {
    linalg::PoolingNcwMaxOp::attachInterface<LinalgMaxPoolReducesLevelModel>(
        *ctx);
  });
}

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
