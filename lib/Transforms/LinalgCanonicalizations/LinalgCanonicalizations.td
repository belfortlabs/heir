#ifndef LIB_TRANSFORMS_LINALGCANONICALIZATIONS_LINALGCANONICALIZATIONS_TD_
#define LIB_TRANSFORMS_LINALGCANONICALIZATIONS_LINALGCANONICALIZATIONS_TD_

include "mlir/Pass/PassBase.td"

def LinalgCanonicalizations : Pass<"linalg-canonicalizations"> {
  let summary = "This pass canonicalizes the linalg.transpose operation of a constant into a transposed constant.";

  let dependentDialects = [
    "arith::ArithDialect",
    "affine::AffineDialect",
  ];

  let description = [{
    This pass canonicalizes the linalg.transpose operation of a constant into a
    transposed constant.
  }];

  let options = [
    Option<"densifyConvolutions", "densify-convolutions", "bool",
           /*default=*/"true",
           "Rewrite dilated convolutions as undilated ones with zeros "
           "inserted into the filter, and sum pools as convolutions over all "
           "channel pairs. Lowerings that handle dilations and pools "
           "directly turn this off, because the zeros in the filter cost "
           "them as much as any other coefficient.">,
  ];
}

#endif  // LIB_TRANSFORMS_LINALGCANONICALIZATIONS_LINALGCANONICALIZATIONS_TD_
