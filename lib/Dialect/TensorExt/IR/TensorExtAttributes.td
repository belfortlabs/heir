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

def TensorExt_ConvPackingAttr : TensorExt_Attr<"ConvPacking", "conv_packing"> {
  let summary = "How a convolution's filter matrix is shaped and laid out.";
  let description = [{
    A convolution lowers to a plaintext-ciphertext matvec over an expanded
    Toeplitz matrix. Several choices fix that matrix, and they are made when
    layouts are assigned but needed again when the kernel is materialized.
    This attribute carries all of them together, so that both passes read one
    value instead of re-deriving it from separate scalars.

    * `matrixDataType` is the data operand the Toeplitz matrix is built
      against. It already accounts for channel padding, and for a `tensor.pad`
      folded into `padding` below, so a reader needs no further arithmetic.
    * `padding` is the symmetric spatial zero padding folded out of a
      `tensor.pad` and into the convolution's own `padding` parameter.
    * `interchangeRows` says the matrix rows are pixel-shuffled, which reserves
      whole channel blocks and so changes the matrix shape.
    * `absorbedMatrixWidth` is the width the filter's diagonal layout was built
      at, when the data operand's slot packing was absorbed into the matrix
      column space. The columns are ciphertext slots then, so the width is the
      ciphertext size rather than the Toeplitz width. Zero means nothing was
      absorbed, and the Toeplitz width applies.
  }];
  let parameters = (ins
    "::mlir::RankedTensorType":$matrixDataType,
    "int64_t":$padding,
    "bool":$interchangeRows,
    "int64_t":$absorbedMatrixWidth
  );
  let genVerifyDecl = 1;
}

def TensorExt_KernelInfoAttr : TensorExt_Attr<"KernelInfo", "kernel_info"> {
  let summary = "What a value's kernel produced, and the gap it was packed with.";
  let description = [{
    Layout assignment records this on each value it lays out.

    * `resultShape` is the shape the resolved FHE kernel produces, which
      accounts for any expansion, striding or channel padding the kernel adds
      on top of the data-semantic shape.
    * `gapFactor` is the multiplexing gap that shape is packed with. A strided
      convolution multiplies its operand's gap by its stride, so a chain of
      them accumulates, and each conv reads the factor off its data operand.

    Unlike tensor_ext.conv_packing, this describes a value rather than an
    operation, because it is what the next kernel in the chain consumes.
  }];
  let parameters = (ins
    "DenseI64ArrayAttr":$resultShape,
    "int64_t":$gapFactor
  );
  // Printed inline rather than behind a mnemonic alias: it lands on nearly
  // every laid-out value, and keeping the value beside the op is what makes the
  // layout tests readable.
  let genMnemonicAlias = 0;
  let genVerifyDecl = 1;
}

#endif  // LIB_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_TD_
