#ifndef LIB_ANALYSIS_PENDINGRESCALEANALYSIS_PENDINGRESCALEANALYSIS_H_
#define LIB_ANALYSIS_PENDINGRESCALEANALYSIS_PENDINGRESCALEANALYSIS_H_

#include <optional>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

// This analysis should be used after --mlir-to-secret-arithmetic
// but before --secret-distribute-generic
// where a whole secret::GenericOp is assumed

/// Whether a secret Value carries a rescale that has not been applied yet,
/// i.e. whether its scale is above the scale of a value that was just mod
/// reduced.
///
/// The analysis models the "mod reduce before multiplication" placement: an op
/// that increases the multiplicative depth gets operands without a pending
/// rescale, so its result does not depend on the state of its operands. The
/// result has a pending rescale, unless the op also reduces the level, which
/// means that it rescales its own result (`kernel.linear_transform`,
/// `kernel.eval_chebyshev`). All other ops forward the state of their
/// operands; a `mgmt.modreduce` clears it.
///
/// This is not the multiplicative depth: a value of positive depth that was
/// already rescaled has nothing pending.
class PendingRescaleState {
 public:
  PendingRescaleState() : value(std::nullopt) {}
  explicit PendingRescaleState(bool pending) : value(pending) {}
  ~PendingRescaleState() = default;

  bool isInitialized() const { return value.has_value(); }

  /// An uninitialized state has nothing pending.
  bool isPending() const { return value.value_or(false); }

  bool operator==(const PendingRescaleState& rhs) const {
    return value == rhs.value;
  }

  static PendingRescaleState join(const PendingRescaleState& lhs,
                                  const PendingRescaleState& rhs) {
    if (!lhs.isInitialized()) return rhs;
    if (!rhs.isInitialized()) return lhs;
    return PendingRescaleState(lhs.isPending() || rhs.isPending());
  }

  void print(llvm::raw_ostream& os) const {
    if (!isInitialized()) {
      os << "PendingRescaleState(uninitialized)";
      return;
    }
    os << "PendingRescaleState(" << (isPending() ? "pending" : "rescaled")
       << ")";
  }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const PendingRescaleState& state) {
    state.print(os);
    return os;
  }

 private:
  std::optional<bool> value;
};

class PendingRescaleLattice : public dataflow::Lattice<PendingRescaleState> {
 public:
  using Lattice::Lattice;
};

class PendingRescaleAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<PendingRescaleLattice>,
      public SecretnessAnalysisDependent<PendingRescaleAnalysis> {
 public:
  /// `inputsPending` sets the state of the values that enter the computation
  /// fresh: the arguments of a `secret.generic` and the results of ops that
  /// reset the multiplicative depth (bootstraps). It is true when the inputs
  /// are encrypted at a scale that needs a mod reduce before the first
  /// multiplication.
  PendingRescaleAnalysis(DataFlowSolver& solver, bool inputsPending = false)
      : dataflow::SparseForwardDataFlowAnalysis<PendingRescaleLattice>(solver),
        inputsPending(inputsPending) {}
  friend class SecretnessAnalysisDependent<PendingRescaleAnalysis>;

  void setToEntryState(PendingRescaleLattice* lattice) override {
    if (isa<SecretTypeInterface>(
            getElementTypeOrSelf(lattice->getAnchor().getType()))) {
      propagateIfChanged(lattice,
                         lattice->join(PendingRescaleState(inputsPending)));
      return;
    }
    propagateIfChanged(lattice, lattice->join(PendingRescaleState()));
  }

  LogicalResult visitOperation(
      Operation* op, ArrayRef<const PendingRescaleLattice*> operands,
      ArrayRef<PendingRescaleLattice*> results) override;

  void visitExternalCall(
      CallOpInterface call,
      ArrayRef<const PendingRescaleLattice*> argumentLattices,
      ArrayRef<PendingRescaleLattice*> resultLattices) override;

 private:
  bool inputsPending;
};

/// Whether `value` has a pending rescale according to the analysis in
/// `solver`. A value without a state has nothing pending.
bool hasPendingRescale(Value value, DataFlowSolver* solver);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_PENDINGRESCALEANALYSIS_PENDINGRESCALEANALYSIS_H_
