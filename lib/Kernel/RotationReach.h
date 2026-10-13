#ifndef LIB_KERNEL_ROTATIONREACH_H_
#define LIB_KERNEL_ROTATIONREACH_H_

#include <cstdint>

namespace mlir {
namespace heir {
namespace kernel {

// The rotation reach of a cyclic matmul kernel is the largest slot offset its
// rotations read from. Only the first numSlots - reach slots of the kernel's
// result are valid, so a ciphertext must hold the reach plus the whole result.
// Layout propagation uses these to select a kernel, and
// convert-to-ciphertext-semantics uses them to find the valid prefix, so the
// two must agree.

// Rotate-and-reduce over `steps` summation steps with rotation stride
// `period`, as in the bicyclic and tricyclic kernels with a plaintext operand.
inline int64_t getRotateAndReduceReach(int64_t period, int64_t steps) {
  return period * (steps - 1);
}

// The BSGS bicyclic ciphertext-ciphertext kernel for (m x n) * (n x p).
inline int64_t getBicyclicMatmulReach(int64_t m, int64_t n, int64_t p) {
  return n * p - 1 + m * (n - 1);
}

// The BSGS tricyclic ciphertext-ciphertext kernel for
// (h x m x n) * (h x n x p).
inline int64_t getTricyclicBatchMatmulReach(int64_t h, int64_t m, int64_t n,
                                            int64_t p) {
  return h * n * p - 1 + h * m * (n - 1);
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_ROTATIONREACH_H_
