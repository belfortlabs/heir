#ifndef LIB_DIALECT_KERNEL_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_KERNEL_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def PrepareLinearTransforms : Pass<"prepare-linear-transforms", "ModuleOp"> {
  let summary = "Split kernel.linear_transform into prepare and apply";
  let description = [{
    Splits each `kernel.linear_transform` into a
    `kernel.prepare_linear_transform` of its diagonals and a
    `kernel.apply_linear_transform` of the resulting prepared transform, when
    the backend target supports `hasKernelLinearTransform`. Only applies to
    inputs matching `SecretLikeTypeInterface` or containers thereof, i.e.,
    LWECiphertextType or lower-level ciphertext types.

    The produced `kernel.prepare_linear_transofrm` op implements
    `PlaintextEncodeOpInterface`, so the later `split-preprocessing` pass
    hoists it into the preprocessing helper functions.

    The prepared type records the level, slot count, and BSGS split baked into
    the encoded diagonals, taken here from the consuming ciphertext's type.
    Backends currently read these off the ciphertext at runtime, where a
    mismatch silently evaluates a wrongly-scaled transform; carrying them in
    the type makes a mismatch a verifier error instead.
  }];
  let dependentDialects = ["mlir::heir::kernel::KernelDialect"];
}

def ImplementLinearTransforms : Pass<"implement-linear-transforms", "ModuleOp"> {
  let summary = "Lower kernel.block_linear_transform to rotations and multiplications";
  let description = [{
    Lowers each `kernel.block_linear_transform` to a baby-step giant-step
    evaluation over all of its blocks at once: each input ciphertext is rotated
    once per distinct baby step, and each partial sum once per distinct
    (output ciphertext, giant step). The giant step is the power of two that
    minimizes the number of ciphertext rotations for the op's set of shifts.

    By default the evaluation is a loop over constant tables, so the IR size
    does not grow with the number of diagonals; `unroll` emits straight-line
    code instead.

    When the backend target supports `hasKernelLinearTransform`, the op is
    instead split into one `kernel.linear_transform` per (output, input) block,
    and the backend evaluates each block natively.
  }];
  let dependentDialects = [
    "mlir::heir::kernel::KernelDialect",
    "mlir::heir::tensor_ext::TensorExtDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
  ];
  let options = [
    Option<"unroll", "unroll", "bool", /*default=*/"false",
           "Emit straight-line code instead of loops.">,
  ];
}

#endif  // LIB_DIALECT_KERNEL_TRANSFORMS_PASSES_TD_
