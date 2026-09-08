#ifndef LIB_DIALECT_CHEDDAR_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_CHEDDAR_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def CheddarElideOutParamCopies : Pass<"cheddar-elide-out-param-copies", "::mlir::func::FuncOp"> {
  let summary = "Write temporaries that are copied into an out-param directly into it";
  let description = [{
    After `buffer-results-to-out-params`, a function that returns e.g. the
    result of a call still allocates a temporary and copies it into the
    `bufferize.result` out-param. Cheddar payloads are move-only and copies
    are expensive GPU deep copies, so replace the temporary with the out-param
    whenever the copy is its last use.
  }];
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::memref::MemRefDialect"];
}

#endif  // LIB_DIALECT_CHEDDAR_TRANSFORMS_PASSES_TD_
