#ifndef LIB_TRANSFORMS_EQUALIZEACTIVATIONRANGES_EQUALIZEACTIVATIONRANGES_TD_
#define LIB_TRANSFORMS_EQUALIZEACTIVATIONRANGES_EQUALIZEACTIVATIONRANGES_TD_

include "mlir/Pass/PassBase.td"

def EqualizeActivationRanges : Pass<"equalize-activation-ranges", "ModuleOp"> {
  let summary = "Fold calibrated activation scales into adjacent linear ops";
  let description = [{
    Rescales the intermediate values of a linalg program so that the
    approximation domains of its range-annotated activations lie within
    `[-target, target]`, without adding any runtime operations. CKKS
    bootstrapping is only accurate for messages of small magnitude, and a
    program with composite-sign ReLUs bootstraps their inputs.

    Every secret floating-point SSA value `x` is replaced by `x / s` for a
    positive factor `s`. Ops that commute with a positive scale, such as ReLU,
    additions, pooling, padding with zeros and reshapes, force their operands
    and results to share a factor. A contraction or convolution with a
    constant weight separates the factor of its input from that of its result,
    and absorbs the transition into its constants: `W' = W * s_in / s_out`
    for the weights and `b' = b / s_out` for the accumulator it is
    initialized with. Constants added to a scaled value are divided by its
    factor. Function arguments, returned values and operands of any other op
    keep factor 1.

    A ReLU whose class is pinned (e.g. it feeds the function result) keeps
    its domain.

    Constants stored as dense resources stay dense resources. A
    `debug.validate` of a rescaled value gets the factor as a `debug.scale`
    attribute.
  }];
  let options = [
    Option<"target", "target", "double", "0.5",
           "Magnitude that the calibrated activation inputs are scaled to">,
    Option<"margin", "margin", "double", "2.0",
           "Factor by which the frontend widened the calibrated ranges about "
           "their centre into the domain attributes">,
  ];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_TRANSFORMS_EQUALIZEACTIVATIONRANGES_EQUALIZEACTIVATIONRANGES_TD_
