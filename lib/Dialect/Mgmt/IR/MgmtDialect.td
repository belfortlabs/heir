#ifndef LIB_DIALECT_MGMT_IR_MGMTDIALECT_TD_
#define LIB_DIALECT_MGMT_IR_MGMTDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Mgmt_Dialect : Dialect {
  let name = "mgmt";
  let description = [{
    The `mgmt` dialect contains scheme-agnostic ciphertext management ops
    (like relinearize and mod reduce), to enable initial high-level compiler
    passes to perform a first pass at parameter selection, while lower-level
    passes may refine them with scheme-specific information.
  }];

  let cppNamespace = "::mlir::heir::mgmt";

  let extraClassDeclaration = [{
    /// Name of the attribute holding MgmtAttr
    constexpr const static ::llvm::StringLiteral
        kArgMgmtAttrName = "mgmt.mgmt";

    constexpr const static ::llvm::StringLiteral
        kMulHeadroomAttrName = "mgmt.mul_headroom";

    /// Module attribute pinning the level fresh ciphertexts and bootstrap
    /// outputs are materialized at: the top of the modulus chain the program
    /// is placed on. Without it the top is the program's own depth. Set by
    /// parameter generation when the chain is deeper than the program (a
    /// chain taken from a parameter file, whose bootstrap lands at a fixed
    /// level), and honored by every annotate-mgmt run after it.
    constexpr const static ::llvm::StringLiteral
        kTopLevelAttrName = "mgmt.top_level";

    /// Name of the attribute holding OpenfheParamsAttr
    constexpr const static ::llvm::StringLiteral
        kArgOpenfheParamsAttrName = "mgmt.openfhe_params";
  }];


  let useDefaultAttributePrinterParser = 1;
}

#endif  // LIB_DIALECT_MGMT_IR_MGMTDIALECT_TD_
