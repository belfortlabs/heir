#ifndef LIB_DIALECT_KERNEL_IR_KERNELOPS_TD_
#define LIB_DIALECT_KERNEL_IR_KERNELOPS_TD_

include "KernelDialect.td"
include "KernelTypes.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "lib/Dialect/HEIRInterfaces.td"

class Kernel_Op<string mnemonic, list<Trait> traits = []> :
    Op<Kernel_Dialect, mnemonic, traits # [Pure]> {
  let cppNamespace = "::mlir::heir::kernel";
}

def Kernel_EvalChebyshevOp : Kernel_Op<"eval_chebyshev", [
  Pure,
  DeclareOpInterfaceMethods<ReducesLevelOpInterface, ["getLevelsToDrop", "getOperandsToReduce"]>,
  ElementwiseMappable,
  IncreasesMulDepthOpInterface
]> {
  let summary = "Evaluates a Chebyshev polynomial.";
  let description = [{
    The `kernel.eval_chebyshev` operation evaluates a Chebyshev polynomial
    on a given input. The domain is required to be `[-1, 1]`.
  }];

  let arguments = (ins
    AnyType:$input,
    ArrayAttr:$coefficients
  );

  let results = (outs
    AnyType:$output
  );

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def Kernel_MaxPoolOp : Kernel_Op<"max_pool", [
  Pure,
  IncreasesMulDepthOpInterface
]> {
  let summary = "Max pools a sequence that fills the first slots of one ciphertext.";
  let description = [{
    The `kernel.max_pool` operation is a 1-D max pool with PyTorch
    `MaxPool1d` semantics and no padding. The input sequence is the first
    `input_length` slots of one ciphertext with `num_slots` slots. The output
    holds the maximum of window `w` in slot `w`, for every window, and zero in
    the other slots.

    The slots at and after `input_length` may hold any values in
    `[-value_bound, value_bound]`, for example a periodic copy of the
    sequence, if no window reads them. This is the case when `input_length` is
    a multiple of `stride` and `ceil_mode` is false.

    `num_slots` is the slot count of the ciphertext. It can be less than half
    the ring degree, so the op keeps it after `--secret-to-ckks` converts the
    cleartext tensor to an LWE type.

    Window `w` reads the slots `w * stride + j * dilation`, for
    `0 <= j < window_size`, that are less than `input_length`. With
    `ceil_mode`, a last window that starts inside the input but extends past
    it is kept; without it, the op drops that window.

    The slot values must lie in `[-value_bound, value_bound]`. Cyclops scales
    them into the domain of its comparison polynomial, and pads the windows
    that it gathers with `-value_bound`, which does not change a maximum.

    The op lowers to one call of Cyclops' `MaxPool<word>::EvaluateMax`, with
    compact output, a padding value of `-value_bound` and no league level cap.
    That call bootstraps only its comparisons, so the result level is the input
    level less `getLevelsToDrop()`. Cyclops folds the scaling for a
    `value_bound` other than 0.5 into the first gather and compaction stages.
    A pool without gather stages, or without compaction stages, uses one more
    level for each scaling that it cannot fold. The input must have at least
    `getRequiredInputLevels()` levels left, which can be more than the op
    drops.

    The level costs come from the Cyclops planner library. So that this
    dialect does not depend on Cyclops, it only promises
    `ReducesLevelOpInterface` and `BootstrapsInternallyOpInterface` for this
    op. The Cheddar target attaches them, see
    `lib/Target/Cheddar/MaxPoolInterfaces.h`.

    Example:

    ```mlir
    %0 = kernel.max_pool %ct {num_slots = 1024 : i64, input_length = 74 : i64, window_size = 2 : i64, stride = 2 : i64, value_bound = 4.0 : f64}
        : tensor<1x1024xf32> -> tensor<1x1024xf32>
    ```
  }];

  let arguments = (ins
    AnyType:$input,
    I64Attr:$num_slots,
    I64Attr:$input_length,
    I64Attr:$window_size,
    I64Attr:$stride,
    DefaultValuedAttr<I64Attr, "1">:$dilation,
    DefaultValuedAttr<BoolAttr, "false">:$ceil_mode,
    F64Attr:$value_bound
  );

  let results = (outs
    AnyType:$output
  );

  let hasVerifier = 1;
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def Kernel_LinearTransformOp : Kernel_Op<"linear_transform", [
  Pure,
  DeclareOpInterfaceMethods<ReducesLevelOpInterface, ["getLevelsToDrop", "getOperandsToReduce"]>,
  IncreasesMulDepthOpInterface
]> {
  let summary = "Performs a linear transform (matrix-vector multiplication) on a ciphertext.";
  let description = [{
    Performs a linear transform on a ciphertext using the diagonal method. Row
    k of the `diagonals` operand is the generalized diagonal named by
    `diagonal_indices[k]`. If `source_row_indices` is present, that diagonal is
    instead read from row `source_row_indices[k]`.
  }];

  let arguments = (ins
    AnyType:$input,
    // The generalized diagonals of the matrix, as a value rather than an
    // attribute: they are often produced by preprocessing (packing the weights
    // once, ahead of any query) rather than folded at compile time. A consumer
    // that needs the values at compile time can recover them with m_Constant.
    // This also matches the backend ops this lowers to, whose diagonals are
    // likewise operands.
    AnyType:$diagonals,
    DenseI64ArrayAttr:$diagonal_indices,
    OptionalAttr<DenseI64ArrayAttr>:$source_row_indices,
    OptionalAttr<F64Attr>:$bsgs_ratio
  );

  let results = (outs
    AnyType:$output
  );

  let assemblyFormat = "$input `,` $diagonals attr-dict `:` type($input) `,` type($diagonals) `->` type($output)";
  let hasVerifier = 1;
}

def Kernel_PrepareLinearTransformOp : Kernel_Op<"prepare_linear_transform", [
  Pure,
  PlaintextEncodeOpInterface
]> {
  let summary = "Prepares (encodes) a linear transform for later application.";
  let description = [{
    Encodes the generalized diagonals of a matrix into a backend-specific
    prepared transform, at the level, slot count, and BSGS split recorded in
    the result type. The prepared transform is applied to a ciphertext with
    `kernel.apply_linear_transform`.

    Preparation is pure cleartext work (row selection, float conversion,
    plaintext encoding), so the op implements PlaintextEncodeOpInterface and
    split-preprocessing hoists it out of the hot path: an inference then only
    pays for the application.

    As with `kernel.linear_transform`, row k of the `diagonals` operand is the
    generalized diagonal named by `diagonal_indices[k]`. If
    `source_row_indices` is present, that diagonal is instead read from row
    `source_row_indices[k]`.
  }];

  let arguments = (ins
    AnyType:$diagonals,
    DenseI64ArrayAttr:$diagonal_indices,
    OptionalAttr<DenseI64ArrayAttr>:$source_row_indices
  );

  let results = (outs
    Kernel_PreparedLinearTransform:$prepared
  );

  let assemblyFormat = "$diagonals attr-dict `:` type($diagonals) `->` type($prepared)";
  let hasVerifier = 1;
}

def Kernel_ApplyLinearTransformOp : Kernel_Op<"apply_linear_transform", [
  Pure,
  DeclareOpInterfaceMethods<ReducesLevelOpInterface, ["getLevelsToDrop", "getOperandsToReduce"]>,
  IncreasesMulDepthOpInterface
]> {
  let summary = "Applies a prepared linear transform to a ciphertext.";
  let description = [{
    Evaluates a transform prepared by `kernel.prepare_linear_transform` on a
    ciphertext. The prepared type's parameters are verified against the
    ciphertext: applying at a level other than the one the diagonals were
    encoded at would silently compute with a wrongly-scaled transform, so a
    mismatch is a compile-time error here rather than a runtime surprise.

    `diagonal_indices` and `diagonal_width` retain planning metadata needed by
    target lowerings after preparation and application are split across
    functions.
  }];

  let arguments = (ins
    AnyType:$input,
    Kernel_PreparedLinearTransform:$prepared,
    OptionalAttr<DenseI64ArrayAttr>:$diagonal_indices,
    OptionalAttr<I64Attr>:$diagonal_width
  );

  let results = (outs
    AnyType:$output
  );

  let assemblyFormat = "$input `,` $prepared attr-dict `:` type($input) `,` type($prepared) `->` type($output)";
  let hasVerifier = 1;
}

#endif  // LIB_DIALECT_KERNEL_IR_KERNELOPS_TD_
