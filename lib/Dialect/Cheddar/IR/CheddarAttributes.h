#ifndef LIB_DIALECT_CHEDDAR_IR_CHEDDARATTRIBUTES_H_
#define LIB_DIALECT_CHEDDAR_IR_CHEDDARATTRIBUTES_H_

#include <cstdint>
#include <optional>
#include <utility>

// IWYU pragma: begin_keep
#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "llvm/include/llvm/ADT/SmallVector.h"       // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/Cheddar/IR/CheddarAttributes.h.inc"

namespace mlir {
namespace heir {
namespace cheddar {

// Module attributes generate-param-ckks records when the modulus chain comes
// from a CHEDDAR parameter file rather than from HEIR's own prime generation:
// the runtime parameter set and, for programs that bootstrap, the bootstrap
// split. cheddar-configure-crypto-context consumes both.
constexpr ::llvm::StringLiteral kParameterSetAttrName = "cheddar.parameter_set";
constexpr ::llvm::StringLiteral kBootstrapConfigAttrName =
    "cheddar.bootstrap_config";

// Module attribute naming the word width of the generated C++ (32 or 64),
// recorded by cheddar-configure-crypto-context for the EmitC entry interface.
constexpr ::llvm::StringLiteral kWordBitsAttrName = "cheddar.word_bits";

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CHEDDAR_IR_CHEDDARATTRIBUTES_H_
