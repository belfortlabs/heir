#ifndef LIB_TRANSFORMS_LINEARKERNEL_RELUNORMALIZATION_TD_
#define LIB_TRANSFORMS_LINEARKERNEL_RELUNORMALIZATION_TD_

include "mlir/Pass/PassBase.td"

def ReluCalibration : Pass<"relu-calibration", "ModuleOp"> {
  let summary = "Make a function return the input ranges of its ReLUs";
  let description = [{
  Prototype. Runs on a plain tensor program (after `secret-forget-secrets`).
  Every function with at least one ReLU `arith.maximumf %x, 0` on a tensor is
  changed to return one 1-D tensor of statistics in place of its results. Its
  first half holds the maximum of `|x|` per channel for each ReLU in program
  order, its second half the mean of `|x|` per channel in the same order. A
  rank-4 tensor has one channel per index of dimension 1; any other tensor is
  one channel.

  Running the function on calibration inputs and combining the vectors gives
  the file that `relu-normalization` reads.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def ReluNormalization : Pass<"relu-normalization", "ModuleOp"> {
  let summary = "Scale ReLU inputs to [-1, 1] and choose an approximation per ReLU";
  let description = [{
  Prototype. Reads the statistics written for `relu-calibration` and rewrites
  each ReLU `max(x, 0)` to `s * max(x * (1 / s), 0)`, where `s` is the
  calibrated bound of `|x|` times `margin`, per channel or per tensor. The new
  ReLU gets the domain [-1, 1] and the attributes `degree` and `relu_method`
  that `polynomial-approximation` reads.

  With `rung=auto` the approximation is the cheapest of a single polynomial
  of degree 5, 15, 31 or 63 and the composite-sign method whose worst-case
  error on [-1, 1] is at most `tau * r`, where `r` is the median over channels
  of `mean |x| / s`.

  One line per ReLU is printed to stderr:
  `relu-normalization <index> <channels> <r> <min s> <max s> <rung>`.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect"];
  let options = [
    Option<"calibrationFile", "calibration-file", "std::string", "\"\"",
           "File with the calibration statistics.">,
    Option<"mode", "mode", "std::string", "\"per-channel\"",
           "per-channel or per-tensor bounds.">,
    Option<"margin", "margin", "double", "1.25",
           "Factor applied to the calibrated bound.">,
    Option<"rung", "rung", "std::string", "\"auto\"",
           "auto, deg5, deg15, deg31, deg63 or composite.">,
    Option<"tau", "tau", "double", "0.1",
           "Largest approximation error relative to the typical input size.">,
    Option<"rungErrors", "rung-errors", "std::string", "\"\"",
           "Worst-case errors on [-1, 1] of deg5, deg15, deg31, deg63 and "
           "composite, separated by colons.">,
  ];
}

#endif  // LIB_TRANSFORMS_LINEARKERNEL_RELUNORMALIZATION_TD_
