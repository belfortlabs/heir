#ifndef LIB_TRANSFORMS_LINEARKERNEL_COUNTOPS_TD_
#define LIB_TRANSFORMS_LINEARKERNEL_COUNTOPS_TD_

include "mlir/Pass/PassBase.td"

def CountOps : Pass<"count-ops", "ModuleOp"> {
  let summary = "Print how often each operation of a function is executed";
  let description = [{
  Prints, for every function with a body, the number of times each kind of
  operation is executed by one call of the function. The module is not changed.

  - An operation inside `affine.for` or `scf.for` loops with constant bounds is
    counted once per iteration.
  - An operation whose result is a tensor of non-builtin elements (for example
    a tensor of ciphertexts) is counted once per element, which is what the
    elementwise lowering of such an operation executes.
  - A `func.call` adds the counts of its callee.
  - Both branches of an `if` are counted. Loops with bounds that are not
    constant are counted as one iteration and reported as `unknown-loop`.
  - The bodies of `linalg` operations are not entered.

  `ckks.rotate`, `bgv.rotate_cols` and `tensor_ext.rotate` with a static shift
  are additionally reported as `distinct-shifts`, the number of different
  shifts in the function and its callees. `kernel.eval_chebyshev` is also
  reported per polynomial degree.

  The output goes to stderr, one tab-separated line per function and
  operation: `count-ops <function> <operation> <count>`.
  }];
}

#endif  // LIB_TRANSFORMS_LINEARKERNEL_COUNTOPS_TD_
