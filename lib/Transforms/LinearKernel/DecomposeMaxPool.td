#ifndef LIB_TRANSFORMS_LINEARKERNEL_DECOMPOSEMAXPOOL_TD_
#define LIB_TRANSFORMS_LINEARKERNEL_DECOMPOSEMAXPOOL_TD_

include "mlir/Pass/PassBase.td"

def DecomposeMaxPool : Pass<"decompose-max-pool"> {
  let summary = "Rewrite max pooling as window taps and elementwise maxima";
  let description = [{
  Rewrites `linalg.pooling_nchw_max` with static shapes into operations that
  the linear kernel and the polynomial approximation handle:

  - One tap per window position: a `linalg.generic` that copies the element
    at that position of every window. Each tap is linear in the input.
  - A balanced tree of maxima over the taps, where `max(a, b)` is written
    `b + max(a - b, 0)` so that each maximum is a ReLU.

  The input may be a `tensor.pad` whose padding value is the identity of the
  maximum (negative infinity or the lowest float). The taps then read a zero
  padded tensor, and a tap that reads padding at some window is replaced
  there by a tap that never reads padding, which leaves the maximum
  unchanged. This is done with one cleartext mask per tap.

  `domain_lower` and `domain_upper` attributes of the pooling operation
  describe the range of its input. The maxima get the range of a difference
  of two such values.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_TRANSFORMS_LINEARKERNEL_DECOMPOSEMAXPOOL_TD_
