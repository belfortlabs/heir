#ifndef LIB_DIALECT_CHEDDAR_IR_CHEDDARATTRIBUTES_TD_
#define LIB_DIALECT_CHEDDAR_IR_CHEDDARATTRIBUTES_TD_

include "CheddarDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Cheddar_Attribute<string name, string attrMnemonic>
    : AttrDef<Cheddar_Dialect, name> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";
}

def Cheddar_BootstrapConfigAttr
    : Cheddar_Attribute<"BootstrapConfig", "bootstrap_config"> {
  let summary = "How the bootstrap circuit splits the levels above the chain";
  let description = [{
    The inputs of the runtime's `BootParameter`: the CoeffToSlot and
    SlotToCoeff level budgets (a depth/rotation trade-off, cf. OpenFHE's
    level-budget-encode/decode), the number of EvalMod levels the
    approximation consumes, and the message headroom `logMessageRatio`
    (~ log2(q0 / scale); the runtime's default when omitted).

    The bootstrap starts at the top of the modulus chain and lands
    `numCtsLevels + numEvalModLevels + numStcLevels` levels lower, at the
    chain's default encryption level.
  }];
  let parameters = (ins
    "int64_t":$numCtsLevels,
    "int64_t":$numStcLevels,
    "int64_t":$numEvalModLevels,
    OptionalParameter<"std::optional<int64_t>">:$logMessageRatio
  );
  let genVerifyDecl = 1;
}

def Cheddar_ParameterSetAttr
    : Cheddar_Attribute<"ParameterSet", "parameter_set"> {
  let summary = "The inputs of a CHEDDAR Parameter object";
  let description = [{
    Everything the runtime's `Parameter<word>` is built from, in the runtime's
    own terms: the ring (`logN`), the base scale (`logScale`), the prime pools
    (`mainPrimes`, `terminalPrimes`, `auxPrimes`) and the modulus chain as a
    `levelConfig` of `(numMain, numTerminal)` pairs, one per level from level 0
    up, each naming how many main and terminal primes that level is built from.
    Consecutive levels need not be prefixes of each other; a 32-bit word chain
    spends two or three primes per level and moves terminal primes in and out.

    Everything but the ring, the scale and the main/aux pools is optional and
    defaults to what HEIR's generated 64-bit chains use: one main prime per
    level (`levelConfig = (1,0), (2,0), ...`), no terminal primes, 64-bit words,
    fresh encryption at the top level, and the runtime's own defaults for the
    secret-key hamming weights.

    `defaultEncryptionLevel` is the level fresh ciphertexts and bootstrap
    outputs sit at; the levels above it (if any) belong to the bootstrap
    circuit.
  }];
  let parameters = (ins
    "int64_t":$logN,
    "int64_t":$logScale,
    "DenseI64ArrayAttr":$mainPrimes,
    "DenseI64ArrayAttr":$auxPrimes,
    OptionalParameter<"DenseI64ArrayAttr">:$terminalPrimes,
    OptionalParameter<"DenseI64ArrayAttr">:$levelConfig,
    OptionalParameter<"std::optional<int64_t>">:$wordBits,
    OptionalParameter<"std::optional<int64_t>">:$defaultEncryptionLevel,
    OptionalParameter<"DenseI64ArrayAttr">:$additionalBase,
    OptionalParameter<"std::optional<int64_t>">:$denseHammingWeight,
    OptionalParameter<"std::optional<int64_t>">:$sparseHammingWeight
  );
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    // The word width the primes are meant for (64 unless stated).
    int64_t getWordBitsOrDefault() const;
    // The (numMain, numTerminal) pair of every level, level 0 first; one main
    // prime per level unless a levelConfig is given.
    ::llvm::SmallVector<std::pair<int64_t, int64_t>> getLevelPairs() const;
    int64_t getMaxLevel() const;
    // The level fresh ciphertexts sit at: the top level unless stated.
    int64_t getDefaultEncryptionLevelOrDefault() const;
    std::pair<int64_t, int64_t> getAdditionalBasePair() const;
  }];
}

#endif  // LIB_DIALECT_CHEDDAR_IR_CHEDDARATTRIBUTES_TD_
