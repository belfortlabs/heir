#ifndef LIB_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_TD_
#define LIB_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_TD_

include "lib/Dialect/TensorExt/IR/TensorExtDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/TensorEncoding.td"

class TensorExt_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TensorExt_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";
  let genMnemonicAlias = 1;
}

def TensorExt_LayoutAttr : TensorExt_Attr<"Layout", "layout"> {
  let summary = "a relation describing the layout of a data-semantic tensor.";
  let description = [{
    This attribute describes how a data-semantic tensor is laid out among a
    tensor of ciphertexts. The layout is described by an integer relation $(d,
    s)$, where $d$ is a multiset of data-semantic tensor indices and $s$ is a
    multiset of slot indices (or coefficient indices). The slot indices are
    defined by two indices: the ciphertext index and the slot index in that
    order. The elements of the relation are defined by a set of quasi-affine
    constraints.

    I.e., a point $((2, 3), (7, 0))$ in the relation says that the data entry
    at index $(2, 3)$ is placed in slot 0 of ciphertext 7. This could be
    defined as part of the relation by a constraint like `row + col + 2 - ct +
    slot = 0`.

    The attribute stores a string representation of the integer relation,
    which follows the ISL syntax for `isl_basic_map`. For example:

    ```mlir
    #vec_layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 1024 = 7 and i0 >= 0 and 0 >= i0 and slot >= 0 and 1023 >= slot and ct = 0 }">

    #mat_layout = #tensor_ext.layout<"{ [row, col] -> [ct, slot] : (slot - row) mod 512 = 0 and (ct + slot - col) mod 512 = 0 and row >= 0 and col >= 0 and ct >= 0 and slot >= 0 and 1023 >= slot and 511 >= ct and 511 >= row and 511 >= col }">

    // Example with local (existential) variables.
    #layout = #tensor_ext.layout<"{[d0] -> [ct, slot] : exists d3, d4 : -ct + d3 = 0 and d0 - d4 * 1024 = 0 and -d0 + 31 >= 0 and d0 >= 0 and ct >= 0 and slot >= 0 and -slot + 1023 >= 0 and -d0 + d3 * 1024 + 1023 >= 0 and d0 - d3 * 1024 >= 0 and -d0 + d4 * 1024 + 1023 >= 0 and d0 - d4 * 1024 >= 0 }">
    ```
  }];
  let parameters = (ins "mlir::StringAttr":$layout);
  let genVerifyDecl = 1;
  let assemblyFormat = "`<` $layout `>`";

  let builders = [
    AttrBuilder<(ins "std::string":$layoutStr), [{
      return $_get($_ctxt, StringAttr::get($_ctxt, layoutStr));
    }]>,
  ];

  let extraClassDeclaration = [{
    presburger::IntegerRelation getIntegerRelation() const;

    std::string getLayoutStr() const {
      return getLayout().str();
    }

    static LayoutAttr getFromIntegerRelation(
      ::mlir::MLIRContext* context, const presburger::IntegerRelation &relation);

    static LayoutAttr composeLayouts(::mlir::ArrayAttr arrayAttr, ::mlir::MLIRContext* context);
  }];
}

def TensorExt_StridedLayoutAttr : TensorExt_Attr<"StridedLayout", "strided_layout"> {
  let summary = "a layout given by a position formula with strides.";
  let description = [{
    This attribute describes the same thing as `tensor_ext.layout`, how a
    data-semantic tensor is laid out among a tensor of ciphertexts, for the
    family of layouts in which the element at index `idx` sits at the global
    position

    ```
    p(idx) = offset + sum over digits of stride * ((idx[axis] / div) mod m)
    ```

    which is slot `p mod slots` of ciphertext `p / slots`. Each digit is
    written `(axis, div, m, stride)`, and `m = 0` means no reduction. With a
    nonzero `period` the tensor occupies one ciphertext and the element is
    also at every position `p + k * period`.

    The family contains row-major layouts, the gapped (channel-multiplexed)
    layouts that strided convolutions and pools produce, and cyclically
    repeated vectors. Unlike a relation, the position of an element and the
    rotation between two layouts are computed in closed form.

    ```mlir
    // A 3x4x5 tensor in row-major order.
    #row_major = #tensor_ext.strided_layout<shape = [3, 4, 5], slots = 64,
        digits = [(0, 1, 0, 20), (1, 1, 0, 5), (2, 1, 0, 1)]>

    // 8 channels of 4x4 on an 8x8 grid with gap 2: four channels fill each
    // 2x2 cell of the grid, the other four go to a second ciphertext.
    #gapped = #tensor_ext.strided_layout<shape = [8, 4, 4], slots = 64,
        digits = [(1, 1, 0, 16), (2, 1, 0, 2),
                  (0, 1, 2, 1), (0, 2, 2, 8), (0, 4, 0, 64)]>

    // A vector of 5 values repeated with period 8.
    #periodic = #tensor_ext.strided_layout<shape = [5], slots = 32,
        digits = [(0, 1, 0, 1)], period = 8>
    ```
  }];
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    // Four entries per digit: axis, div, mod, stride.
    ArrayRefParameter<"int64_t">:$digits,
    "int64_t":$slots,
    "int64_t":$offset,
    "int64_t":$period
  );
  let genVerifyDecl = 1;
  let assemblyFormat = ?;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::heir::StridedLayout getStridedLayout() const;

    static StridedLayoutAttr getFromStridedLayout(
      ::mlir::MLIRContext* context, const ::mlir::heir::StridedLayout& layout);

    /// The layout as the relation that `tensor_ext.layout` stores.
    presburger::IntegerRelation getIntegerRelation() const;
  }];
}

def TensorExt_OriginalTypeAttr : TensorExt_Attr<"OriginalType", "original_type"> {
  let summary = "The original type of a secret tensor whose layout has been converted to ciphertext semantics.";
  let description = [{
    This attribute is used to retain the original type of a secret tensor after
    its conversion to ciphertext semantics, i.e. after applying any padding or
    alignment to fill ciphertext data types. For example, if a
    `!secret.secret<tensor<32xi8>>` is laid out in a ciphertext with
    1024 slots, the new type would be `!secret.secret<tensor<1024xi8>>`
    with attribute `tensor_ext.original_type<!secret.secret<tensor<32xi8>>`.
  }];
  let parameters = (ins
    "::mlir::Type":$originalType,
    "::mlir::Attribute":$layout
  );
  let assemblyFormat =  "`<` struct(params) `>`";
}

#endif  // LIB_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_TD_
