#ifndef LIB_KERNEL_LINEARTRANSFORMIMPLEMENTATION_H_
#define LIB_KERNEL_LINEARTRANSFORMIMPLEMENTATION_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Utils/Layout/SparseBsgs.h"

namespace mlir {
namespace heir {
namespace kernel {

/// Options for implementSparseBsgs.
struct SparseBsgsOptions {
  /// Emit straight-line code instead of loops over constant index tables.
  bool unroll = false;
  /// Row `k` of the diagonals operand already holds diagonal `k` rotated right
  /// by the giant part of its shift (see SparseBsgsPlan), so no plaintext
  /// rotation is emitted.
  bool diagonalsPreRotated = false;
};

/// Returns a DAG that evaluates a block linear transform
///
///   y[key.ctOut] += diagonals[k] * rotateLeft(x[key.ctIn], key.shift)
///
/// over the keys the plan was built from, in the plan's baby-step giant-step
/// order.
///
/// `input` has shape {numIn, N}, `diagonals` has shape {numKeys, N}, and the
/// result has shape {numOut, N}. `rowType` is the type of one row of `input`.
/// Output rows no key writes to are zero.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementSparseBsgs(const SparseBsgsPlan& plan, const T& input,
                    const T& diagonals, int64_t numOut, const DagType& rowType,
                    const SparseBsgsOptions& options = {}) {
  using NodeTy = ArithmeticDagNode<T>;
  using NodePtr = std::shared_ptr<NodeTy>;

  NodePtr inputDag = NodeTy::leaf(input);
  NodePtr diagonalsDag = NodeTy::leaf(diagonals);
  int64_t numSlots = input.getShape().back();
  int64_t numBaby = plan.babyCt.size();
  int64_t numRows = plan.row.size();
  NodePtr emptyOutput = NodeTy::empty({numOut, numSlots}, rowType);

  if (options.unroll) {
    std::vector<NodePtr> baby;
    baby.reserve(numBaby);
    for (int64_t t = 0; t < numBaby; ++t) {
      NodePtr ct = NodeTy::extract(inputDag, plan.babyCt[t]);
      baby.push_back(plan.babyShift[t] == 0
                         ? ct
                         : NodeTy::leftRotate(ct, plan.babyShift[t]));
    }

    std::vector<NodePtr> outputs(numOut, nullptr);
    NodePtr part = nullptr;
    for (int64_t r = 0; r < numRows; ++r) {
      NodePtr diagonal = NodeTy::extract(diagonalsDag, plan.row[r]);
      if (!options.diagonalsPreRotated && plan.giant[r] != 0) {
        diagonal = NodeTy::leftRotate(diagonal, -plan.giant[r]);
      }
      NodePtr term = NodeTy::mul(diagonal, baby[plan.babyOf[r]]);
      part = part == nullptr ? term : NodeTy::add(part, term);
      if (!plan.isLast[r]) continue;
      if (plan.giant[r] != 0) part = NodeTy::leftRotate(part, plan.giant[r]);
      NodePtr& output = outputs[plan.out[r]];
      output = output == nullptr ? part : NodeTy::add(output, part);
      part = nullptr;
    }

    NodePtr result = emptyOutput;
    for (int64_t o = 0; o < numOut; ++o) {
      NodePtr row =
          outputs[o] == nullptr ? NodeTy::splat(0, rowType) : outputs[o];
      result = NodeTy::insert(row, result, static_cast<size_t>(o));
    }
    return result;
  }

  // The rolled form reads the plan from constant index tables.
  DagType indexTy = DagType::integer(64);
  auto table = [&](const std::vector<int64_t>& values) {
    std::vector<double> asDouble(values.begin(), values.end());
    return NodeTy::constantTensor(
        std::move(asDouble),
        DagType::intTensor(64, {static_cast<int64_t>(values.size())}));
  };
  std::vector<int64_t> isLast(plan.isLast.begin(), plan.isLast.end());

  // A node is materialized where it is first used, so values that a loop
  // body shares with code outside it are passed in as loop arguments, and
  // each use of a zero row is a node of its own.
  auto zeroRow = [&]() { return NodeTy::splat(0, rowType); };
  NodePtr zeroOutput = NodeTy::resultAt(
      NodeTy::loop(emptyOutput, rowType, /*lower=*/0,
                   /*upper=*/numOut, /*step=*/1,
                   [&](NodePtr o, const NodePtr& acc) {
                     return NodeTy::yield({NodeTy::insert(zeroRow(), acc, o)});
                   }),
      0);
  if (numRows == 0) return zeroOutput;

  NodePtr babySteps = NodeTy::resultAt(
      NodeTy::loop(NodeTy::empty({numBaby, numSlots}, rowType), rowType,
                   /*lower=*/0, /*upper=*/numBaby, /*step=*/1,
                   [&](NodePtr t, const NodePtr& acc) {
                     NodePtr ct = NodeTy::extract(
                         inputDag, NodeTy::extract(table(plan.babyCt), t));
                     NodePtr rotated = NodeTy::leftRotate(
                         ct, NodeTy::extract(table(plan.babyShift), t));
                     return NodeTy::yield({NodeTy::insert(rotated, acc, t)});
                   }),
      0);

  NodePtr mainLoop = NodeTy::loop(
      {zeroOutput, zeroRow(), babySteps}, {rowType, rowType, rowType},
      /*lower=*/0, /*upper=*/numRows, /*step=*/1,
      [&](NodePtr r, const std::vector<NodePtr>& iterArgs) {
        NodePtr output = iterArgs[0];
        NodePtr part = iterArgs[1];
        NodePtr baby = iterArgs[2];
        NodePtr giantShift = NodeTy::extract(table(plan.giant), r);

        NodePtr diagonal =
            NodeTy::extract(diagonalsDag, NodeTy::extract(table(plan.row), r));
        if (!options.diagonalsPreRotated) {
          diagonal = NodeTy::leftRotate(
              diagonal,
              NodeTy::sub(NodeTy::constantScalar(0, indexTy), giantShift));
        }
        NodePtr newPart = NodeTy::add(
            part,
            NodeTy::mul(
                diagonal,
                NodeTy::extract(baby, NodeTy::extract(table(plan.babyOf), r))));

        NodePtr closesGroup = NodeTy::comparison(
            NodeTy::extract(table(isLast), r),
            NodeTy::constantScalar(1, indexTy), ComparisonPredicate::EQ);
        NodePtr step = NodeTy::ifElse(
            closesGroup,
            [&]() {
              NodePtr outIndex = NodeTy::extract(table(plan.out), r);
              NodePtr sum =
                  NodeTy::add(NodeTy::extract(output, outIndex),
                              NodeTy::leftRotate(newPart, giantShift));
              return NodeTy::yield(
                  {NodeTy::insert(sum, output, outIndex), zeroRow()});
            },
            [&]() { return NodeTy::yield({output, newPart}); });
        return NodeTy::yield(
            {NodeTy::resultAt(step, 0), NodeTy::resultAt(step, 1), baby});
      });
  return NodeTy::resultAt(mainLoop, 0);
}

/// Returns a DAG that sums a box of rotations of one ciphertext,
///
///   sum over 0 <= i_k < count_k of rotateLeft(x, sum_k stride_k * i_k),
///
/// as one logarithmic rotate-and-sum per `(stride, count)` axis. This is a
/// linear transform whose coefficients are all one, evaluated without any
/// plaintext multiplication.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementUniformBoxSum(const T& input,
                       const std::vector<std::pair<int64_t, int64_t>>& axes) {
  using NodeTy = ArithmeticDagNode<T>;
  using NodePtr = std::shared_ptr<NodeTy>;
  NodePtr result = NodeTy::leaf(input);
  DagReducer<T> add = [](NodePtr lhs, NodePtr rhs) {
    return NodeTy::add(lhs, rhs);
  };
  for (const auto& [stride, count] : axes) {
    if (count <= 1) continue;
    result =
        implementRotateAndReduceAccumulation<T>(result, stride, count, add);
  }
  return result;
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_LINEARTRANSFORMIMPLEMENTATION_H_
