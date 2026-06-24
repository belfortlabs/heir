#ifndef LIB_DIALECT_CHEDDAR_TRANSFORMS_CONFIGURECRYPTOCONTEXT_TD_
#define LIB_DIALECT_CHEDDAR_TRANSFORMS_CONFIGURECRYPTOCONTEXT_TD_

include "mlir/Pass/PassBase.td"

def CheddarConfigureCryptoContext : Pass<"cheddar-configure-crypto-context"> {
  let summary = "Configure CHEDDAR crypto context from scheme parameters.";

  let description = [{
    This pass reads the `ckks.schemeParam` module attribute and re-exposes the
    CHEDDAR-relevant parameters (`cheddar.logN`, `cheddar.logDefaultScale`,
    `cheddar.Q`, `cheddar.P`) as module attributes, then removes the CKKS
    module attributes.

    It also generates a `<entry>__configure` function that builds the CHEDDAR
    `Parameter` from the modulus chain, creates the `Context` and
    `UserInterface`, and prepares a rotation key for every distinct rotation
    distance used by the program (`cheddar.hrot` / `cheddar.hrot_add`). The
    context and user_interface are handed back to the caller through owning
    out-params (`std::shared_ptr<Context<word>>&` /
    `std::unique_ptr<UserInterface<word>>&`), mirroring the lattigo backend's
    `__configure` so the medusa harness stays a thin wrapper.

    After this pass, the module no longer depends on the CKKS dialect.
  }];

  let options = [
    Option<"entryFunction", "entry-function", "std::string", "", "Entry function to configure (auto-detected if empty)">,
  ];

  let dependentDialects = [
    "mlir::heir::cheddar::CheddarDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::func::FuncDialect",
  ];
}

#endif  // LIB_DIALECT_CHEDDAR_TRANSFORMS_CONFIGURECRYPTOCONTEXT_TD_
