#ifndef LIB_UTILS_MAXPOOLUTILS_H_
#define LIB_UTILS_MAXPOOLUTILS_H_

#include <cstdint>
#include <string>

#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {

// Whether `value` is a constant -inf float.
bool isNegativeInfinity(Value value);

// Whether `init` is a tensor of -inf, either as a linalg.fill or as the splat
// constant that FoldConstantFill turns that fill into.
bool isNegativeInfinityTensor(Value init);

// The shape of a kernel.max_pool: a 1-D max pool over the slots of one
// ciphertext, with the fields of Cyclops' MaxPoolConfig
// (cyclops/include/extension/max/MaxPoolPlanner.h). The output is compact:
// window w's maximum lands in slot w, and the padding value is -valueBound.
struct MaxPoolShape {
  int64_t numSlots;
  int64_t inputLength;
  int64_t windowSize;
  int64_t stride;
  int64_t dilation;
  bool ceilMode;
  // Every input value v satisfies |v| <= valueBound.
  double valueBound;
};

// The shape of a linalg.pooling_ncw_max that lowers to one kernel.max_pool.
struct SupportedMaxPool {
  int64_t channels;
  // The input length per channel, a multiple of the window size.
  int64_t length;
  // The output length per channel, length / windowSize.
  int64_t outputLength;
  int64_t windowSize;
  // A bound on the absolute value of each input element, from the domain
  // [domain_lower, domain_upper]: max(|domain_lower|, |domain_upper|). Cyclops
  // scales the values into its comparison domain with it.
  double valueBound;

  // The kernel.max_pool over the channels laid end to end in one ciphertext
  // of `numSlots` slots. The windows tile each channel exactly, so no window
  // crosses into the next channel.
  MaxPoolShape getKernelShape(int64_t numSlots) const {
    return MaxPoolShape{numSlots,   channels * length, windowSize,
                        windowSize, /*dilation=*/1,    /*ceilMode=*/false,
                        valueBound};
  }
};

// Checks that `op` has the form that lowers to one kernel.max_pool: N = 1,
// stride equal to the window size, dilation 1, an input length that the
// windows tile exactly, a -inf init, and a float domain_lower/domain_upper
// that bounds the input. On failure,
// `reason` says what does not match.
FailureOr<SupportedMaxPool> getSupportedMaxPool(linalg::PoolingNcwMaxOp op,
                                                std::string& reason);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_MAXPOOLUTILS_H_
