#ifndef LIB_TRANSFORMS_LINEARKERNEL_STRIDEDLAYOUTPROPAGATION_TD_
#define LIB_TRANSFORMS_LINEARKERNEL_STRIDEDLAYOUTPROPAGATION_TD_

include "mlir/Pass/PassBase.td"

def StridedLayoutPropagation : Pass<"strided-layout-propagation"> {
  let summary = "Assign strided ciphertext layouts to secret tensors";
  let description = [{
  This pass is an alternative to `layout-propagation` that assigns
  `#tensor_ext.strided_layout` attributes instead of relation-based layouts.

  Each secret tensor argument of a function is laid out in row-major order.
  Layouts are then propagated forward:

  - A structured linalg operation that is linear in its one secret operand
    (convolutions, sum pools, products with cleartext matrices, cleartext
    scaling, broadcasts) gets the layout that keeps its number of diagonals
    low; see `inducedOutputLayout`. Its cleartext operands get no layout.
  - A zero `tensor.pad` whose users are all such operations gets the layout of
    its source and is folded into them. Any other zero pad is itself a linear
    operation.
  - Reshapes, static slices and operations that permute axes (such as
    `linalg.transpose`) move no element when the layout of the result is
    again strided: the result is the operand's ciphertexts under a relabeled
    layout. When it is not, the operation is a linear operation like any
    other and moves the elements into a new layout.
  - `tensor.insert_slice` of a secret tensor keeps the layout of the tensor
    inserted into; the inserted tensor is moved into place. Chains of
    insertions into an empty tensor, which is what `tensor.concat` decomposes
    into, are sums of such moves.
  - Other operations must have operands and results of one shape and are
    applied slot by slot. Secret operands whose layouts differ are converted
    with `tensor_ext.convert_layout` to the layout among them that needs the
    fewest diagonals in total, and cleartext tensor operands get a
    `tensor_ext.assign_layout`.

  Operations outside this set are reported as errors.
  }];
  let options = [
    Option<
      "minSlotCount",
      "min-slot-count",
      "int",
      /*default=*/"1024",
      "Number of slots of a ciphertext."
    >
  ];

  let dependentDialects = [
    "mlir::heir::secret::SecretDialect",
    "mlir::heir::tensor_ext::TensorExtDialect",
  ];
}

#endif  // LIB_TRANSFORMS_LINEARKERNEL_STRIDEDLAYOUTPROPAGATION_TD_
