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

// The bootstrap split a bootstrapping program's key planning needs,
// recorded by cheddar-configure-crypto-context on the Cyclops client setup.
constexpr ::llvm::StringLiteral kBootstrapConfigAttrName =
    "cheddar.bootstrap_config";

// Module attribute naming the word width of the generated C++ (32 or 64),
// recorded by cheddar-configure-crypto-context for the EmitC entry interface.
constexpr ::llvm::StringLiteral kWordBitsAttrName = "cheddar.word_bits";

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CHEDDAR_IR_CHEDDARATTRIBUTES_H_
