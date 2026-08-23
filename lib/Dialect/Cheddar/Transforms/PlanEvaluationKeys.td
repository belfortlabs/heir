#ifndef LIB_DIALECT_CHEDDAR_TRANSFORMS_PLANEVALUATIONKEYS_TD_
#define LIB_DIALECT_CHEDDAR_TRANSFORMS_PLANEVALUATIONKEYS_TD_

include "mlir/Pass/PassBase.td"

def CheddarPlanEvaluationKeys : Pass<"cheddar-plan-evaluation-keys"> {
  let summary = "Resolve Cyclops key requirements into a concrete key list.";

  let description = [{
    Turns the key *requirements* `cheddar-configure-crypto-context` records on
    the Cyclops client setup function into the concrete list of evaluation keys
    the client must generate.

    The requirements say which rotations the program performs, the diagonal
    shape of each linear transform, and the bootstrap configuration. Going from
    those to an actual key set means choosing a baby-step/giant-step split for
    every transform and unfolding the bootstrap's DFT and EvalMod stages, which
    is handled by the linked Cyclops planner library. This pass records the
    resulting key list.

    It reads `cheddar.rotation_keys`, `cheddar.linear_transform_keys` and the
    `cheddar.bootstrap_*` attributes, and replaces them with
    `cheddar.evaluation_keys`: a flat i64 array of
    `(family, rot_idx, level, key_mode, required_num_aux)` tuples, sorted and
    deduplicated so the generated code is reproducible.

    The planner is built from the pinned Cyclops dependency without CUDA.
  }];

  let dependentDialects = [
    "mlir::heir::cheddar::CheddarDialect",
    "mlir::func::FuncDialect",
  ];
}

#endif  // LIB_DIALECT_CHEDDAR_TRANSFORMS_PLANEVALUATIONKEYS_TD_
