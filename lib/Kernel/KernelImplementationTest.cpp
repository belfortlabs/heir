#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/EvalVisitor.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Kernel/KernelName.h"
#include "lib/Kernel/RotationCountVisitor.h"
#include "lib/Utils/Layout/Codegen.h"
#include "lib/Utils/Layout/Evaluate.h"
#include "lib/Utils/Layout/Utils.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {
namespace {

// Parametrize over whether the kernel is unrolled and whether rows are
// interchanged
class KernelImplementationTest
    : public testing::TestWithParam<std::tuple<bool, bool>> {};

TEST_P(KernelImplementationTest, TestHaleviShoupMatvec) {
  std::vector<int> vector = {0, 1, 2, 3};
  // Pre-packed diagonally
  std::vector<std::vector<int>> matrix = {
      {0, 5, 10, 15}, {1, 6, 11, 12}, {2, 7, 8, 13}, {3, 4, 9, 14}};
  std::vector<int> expected = {14, 38, 62, 86};
  LiteralValue matrixInput = matrix;
  LiteralValue vectorInput = vector;

  auto dag = implementHaleviShoup(
      vectorInput, matrixInput, {4, 4}, DagType::intTensor(32, {4}),
      /*zeroDiagonals=*/{}, /*unroll=*/std::get<0>(GetParam()));
  LiteralValue actual = evalKernel(dag)[0];
  EXPECT_EQ(std::get<std::vector<int>>(actual.get()), expected);
}

TEST(KernelImplementationTest, TestHaleviShoupMatvecZeroDiagonals) {
  std::vector<int> vector = {0, 1, 2, 3};
  // Pre-packed diagonally
  // Row 1 is zeroed out (diagonal 1)
  std::vector<std::vector<int>> matrix = {
      {0, 5, 10, 15}, {0, 0, 0, 0}, {2, 7, 8, 13}, {3, 4, 9, 14}};
  std::vector<int> expected = {13, 26, 29, 86};
  LiteralValue matrixInput = matrix;
  LiteralValue vectorInput = vector;

  std::map<int, bool> zeroDiagonals = {{1, true}};

  auto dag = implementHaleviShoup(vectorInput, matrixInput, {4, 4},
                                  DagType::intTensor(32, {4}), zeroDiagonals,
                                  /*unroll=*/false);
  LiteralValue actual = evalKernel(dag)[0];
  EXPECT_EQ(std::get<std::vector<int>>(actual.get()), expected);
}

TEST_P(KernelImplementationTest, TestHaleviShoupMatvecMultipleZeroDiagonals) {
  std::vector<int> vector = {0, 1, 2, 3};
  // Pre-packed diagonally
  // Rows 1 and 3 are zeroed out (diagonals 1 and 3)
  std::vector<std::vector<int>> matrix = {
      {0, 5, 10, 15}, {0, 0, 0, 0}, {2, 7, 8, 13}, {0, 0, 0, 0}};
  std::vector<int> expected = {4, 26, 20, 58};
  LiteralValue matrixInput = matrix;
  LiteralValue vectorInput = vector;

  std::map<int, bool> zeroDiagonals = {{1, true}, {3, true}};

  auto dag = implementHaleviShoup(vectorInput, matrixInput, {4, 4},
                                  DagType::intTensor(32, {4}), zeroDiagonals,
                                  /*unroll=*/std::get<0>(GetParam()));
  LiteralValue actual = evalKernel(dag)[0];
  EXPECT_EQ(std::get<std::vector<int>>(actual.get()), expected);
}

TEST_P(KernelImplementationTest, TestHaleviShoupMatvecSingleNonZeroDiagonal) {
  std::vector<int> vector = {0, 1, 2, 3};
  // Pre-packed diagonally
  // Only diagonal 2 is nonzero
  std::vector<std::vector<int>> matrix = {
      {0, 0, 0, 0}, {0, 0, 0, 0}, {2, 7, 8, 13}, {0, 0, 0, 0}};
  std::vector<int> expected = {4, 21, 0, 13};
  LiteralValue matrixInput = matrix;
  LiteralValue vectorInput = vector;

  std::map<int, bool> zeroDiagonals = {{0, true}, {1, true}, {3, true}};

  auto dag = implementHaleviShoup(vectorInput, matrixInput, {4, 4},
                                  DagType::intTensor(32, {4}), zeroDiagonals,
                                  /*unroll=*/std::get<0>(GetParam()));
  LiteralValue actual = evalKernel(dag)[0];
  EXPECT_EQ(std::get<std::vector<int>>(actual.get()), expected);
}

TEST_P(KernelImplementationTest, HaleviShoup3x5) {
  // Original matrix:
  // [ 0,  1,  2,  3,  4]
  // [ 5,  6,  7,  8,  9]
  // [10, 11, 12, 13, 14]
  //
  // Padded to 4x8:
  // [ 0,  1,  2,  3,  4, 0, 0, 0]
  // [ 5,  6,  7,  8,  9, 0, 0, 0]
  // [10, 11, 12, 13, 14, 0, 0, 0]
  // [ 0,  0,  0,  0,  0, 0, 0, 0]
  //
  // Diagonalized 8x8 matrix:
  std::vector<std::vector<int>> matrix = {{0, 6, 12, 0, 4, 0, 0, 0},
                                          {1, 7, 13, 0, 0, 0, 0, 0},
                                          {2, 8, 14, 0, 0, 0, 10, 0},
                                          {3, 9, 0, 0, 0, 5, 11, 0}};
  // Original vector {0, 1, 2, 3, 4} padded to size 8.
  std::vector<int> vector = {0, 1, 2, 3, 4, 0, 0, 0};
  std::vector<int> expected = {30, 80, 130};

  LiteralValue matrixInput = matrix;
  LiteralValue vectorInput = vector;

  auto dag = implementHaleviShoup(
      vectorInput, matrixInput, {3, 5}, DagType::intTensor(32, {8}),
      /*zeroDiagonals=*/{}, /*unroll=*/std::get<0>(GetParam()));
  LiteralValue result = evalKernel(dag)[0];
  auto actual = std::get<std::vector<int>>(result.get());

  // The result is of size 8, but we only care about the first 3 elements.
  EXPECT_EQ(std::vector<int>(actual.begin(), actual.begin() + 3), expected);
}

TEST_P(KernelImplementationTest, HaleviShoup4x2) {
  // Original matrix (4x2):
  // [0, 1]
  // [2, 3]
  // [4, 5]
  // [6, 7]
  //
  // Pre-packed diagonally:
  // diag_0 = (0, 3, 4, 7)
  // diag_1 = (1, 2, 5, 6)
  std::vector<std::vector<int>> matrix = {{0, 3, 4, 7}, {1, 2, 5, 6}};
  // Replicated vector {3, 4, 3, 4}
  std::vector<int> vector = {3, 4, 3, 4};
  std::vector<int> expected = {4, 18, 32, 46};

  LiteralValue matrixInput = matrix;
  LiteralValue vectorInput = vector;

  auto dag = implementHaleviShoup(
      vectorInput, matrixInput, {4, 2}, DagType::intTensor(32, {4}),
      /*zeroDiagonals=*/{}, /*unroll=*/std::get<0>(GetParam()));
  LiteralValue result = evalKernel(dag)[0];
  auto actual = std::get<std::vector<int>>(result.get());

  EXPECT_EQ(actual, expected);
}

TEST_P(KernelImplementationTest, HaleviShoup8x4) {
  // Original matrix (8x4):
  // [ 0,  1,  2,  3]
  // [ 4,  5,  6,  7]
  // [ 8,  9, 10, 11]
  // [12, 13, 14, 15]
  // [16, 17, 18, 19]
  // [20, 21, 22, 23]
  // [24, 25, 26, 27]
  // [28, 29, 30, 31]
  //
  // Pre-packed diagonally:
  // diag_0 = (0, 5, 10, 15, 16, 21, 26, 31)
  // diag_1 = (1, 6, 11, 12, 17, 22, 27, 28)
  // diag_2 = (2, 7, 8, 13, 18, 23, 24, 29)
  // diag_3 = (3, 4, 9, 14, 19, 20, 25, 30)
  std::vector<std::vector<int>> matrix = {{0, 5, 10, 15, 16, 21, 26, 31},
                                          {1, 6, 11, 12, 17, 22, 27, 28},
                                          {2, 7, 8, 13, 18, 23, 24, 29},
                                          {3, 4, 9, 14, 19, 20, 25, 30}};
  // Replicated vector {3, 4, 5, 6, 3, 4, 5, 6}
  std::vector<int> vector = {3, 4, 5, 6, 3, 4, 5, 6};
  std::vector<int> expected = {32, 104, 176, 248, 320, 392, 464, 536};

  LiteralValue matrixInput = matrix;
  LiteralValue vectorInput = vector;

  auto dag = implementHaleviShoup(
      vectorInput, matrixInput, {8, 4}, DagType::intTensor(32, {8}),
      /*zeroDiagonals=*/{}, /*unroll=*/std::get<0>(GetParam()));
  LiteralValue result = evalKernel(dag)[0];
  auto actual = std::get<std::vector<int>>(result.get());

  EXPECT_EQ(actual, expected);
}

TEST_P(KernelImplementationTest, HaleviShoup5x3) {
  // Original matrix (5x3):
  // [ 0,  1,  2]
  // [ 3,  4,  5]
  // [ 6,  7,  8]
  // [ 9, 10, 11]
  // [12, 13, 14]
  //
  // Padded to 8x4 and packed diagonally:
  // diag_0 = (0, 4, 8, 0, 12, 0, 0, 0)
  // diag_1 = (1, 5, 0, 9, 13, 0, 0, 0)
  // diag_2 = (2, 0, 6, 10, 14, 0, 0, 0)
  // diag_3 = (0, 3, 7, 11, 0, 0, 0, 0)
  std::vector<std::vector<int>> matrix = {{0, 4, 8, 0, 12, 0, 0, 0},
                                          {1, 5, 0, 9, 13, 0, 0, 0},
                                          {2, 0, 6, 10, 14, 0, 0, 0},
                                          {0, 3, 7, 11, 0, 0, 0, 0}};
  // Padded and replicated vector {1, 2, 3, 0, 1, 2, 3, 0}
  std::vector<int> vector = {1, 2, 3, 0, 1, 2, 3, 0};
  std::vector<int> expected = {8, 26, 44, 62, 80};

  LiteralValue matrixInput = matrix;
  LiteralValue vectorInput = vector;

  auto dag = implementHaleviShoup(
      vectorInput, matrixInput, {5, 3}, DagType::intTensor(32, {8}),
      /*zeroDiagonals=*/{}, /*unroll=*/std::get<0>(GetParam()));
  LiteralValue result = evalKernel(dag)[0];
  auto actual = std::get<std::vector<int>>(result.get());

  // The result is of size 8, but we only care about the first 5 elements.
  EXPECT_EQ(std::vector<int>(actual.begin(), actual.begin() + 5), expected);
}

TEST(KernelImplementationTest, TestExtract) {
  std::vector<std::vector<int>> matrix = {
      {0, 5, 10, 15}, {1, 6, 11, 12}, {2, 7, 8, 13}, {3, 4, 9, 14}};
  std::vector<int> expected = {1, 6, 11, 12};
  LiteralValue matrixInput(matrix);

  auto dag = ArithmeticDagNode<LiteralValue>::extract(
      ArithmeticDagNode<LiteralValue>::leaf(matrixInput), 1);
  LiteralValue actual = evalKernel(dag)[0];
  EXPECT_EQ(std::get<std::vector<int>>(actual.get()), expected);
}

TEST(KernelImplementationTest, HaleviShoup512x784WithLayout) {
  MLIRContext context;
  int rows = 512;
  int cols = 784;
  int numSlots = 1024;

  std::vector<int> vector(numSlots, 0);
  for (int i = 0; i < cols; ++i) {
    vector[i] = 1;
  }
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols, 1));

  auto diagonalLayout = getDiagonalLayoutRelation(
      RankedTensorType::get({rows, cols}, mlir::IndexType::get(&context)),
      numSlots);
  std::vector<std::vector<int>> diagonalMatrix =
      evaluateLayoutOnMatrix(diagonalLayout, matrix);

  std::vector<int> expected(numSlots, 784);

  LiteralValue matrixInput = diagonalMatrix;
  LiteralValue vectorInput = vector;

  auto dag = implementHaleviShoup(vectorInput, matrixInput, {rows, cols},
                                  DagType::intTensor(32, {numSlots}),
                                  /*zeroDiagonals=*/{}, /*unroll=*/true);
  LiteralValue actual = evalKernel(dag)[0];
  EXPECT_EQ(std::get<std::vector<int>>(actual.get()), expected);
}

TEST(KernelImplementationTest, BicyclicMatmul) {
  MLIRContext context;
  std::vector<std::vector<int>> matrixA = {
      {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}};
  std::vector<std::vector<int>> matrixB = {
      {1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}};
  int m = 3;
  int n = 5;
  int p = 2;
  int numSlots = m * n * p;

  auto layoutA = getBicyclicLayoutRelation(
      RankedTensorType::get({m, n}, mlir::IndexType::get(&context)), numSlots);
  auto packedA = evaluateLayoutOnMatrix(layoutA, matrixA);

  auto layoutB = getBicyclicLayoutRelation(
      RankedTensorType::get({n, p}, mlir::IndexType::get(&context)), numSlots);
  auto packedB = evaluateLayoutOnMatrix(layoutB, matrixB);

  LiteralValue packedAValue = packedA[0];
  LiteralValue packedBValue = packedB[0];

  auto dag = implementBicyclicMatmul(packedAValue, packedBValue, m, n, p,
                                     DagType::intTensor(32, {numSlots}));
  LiteralValue result = evalKernel(dag)[0];
  auto resultVec = std::get<std::vector<int>>(result.get());

  auto resultLayout = getBicyclicLayoutRelation(
      RankedTensorType::get({m, p}, mlir::IndexType::get(&context)), numSlots);
  auto unpackedResult =
      unpackLayoutToMatrix<int>(resultLayout, {resultVec}, {m, p});

  std::vector<std::vector<int>> expected = {{95, 110}, {220, 260}, {345, 410}};
  EXPECT_EQ(unpackedResult, expected);
}

TEST(KernelImplementationTest, BicyclicCtPtDiagonalMatmul) {
  MLIRContext context;
  std::vector<std::vector<int>> matrixA = {
      {1, 2, 3, 5}, {7, 11, 13, 17}, {19, 23, 29, 31}};
  std::vector<std::vector<int>> matrixB = {{2, 3, 5, 7, 11},
                                           {13, 17, 19, 23, 29},
                                           {31, 37, 41, 43, 47},
                                           {53, 59, 61, 67, 71}};
  int m = 3;
  int n = 4;
  int p = 5;
  int numSlots = 64;

  auto layoutA = getBicyclicLayoutRelation(
      RankedTensorType::get({m, n}, mlir::IndexType::get(&context)), numSlots);
  auto packedA = evaluateLayoutOnMatrix(layoutA, matrixA);

  auto layoutB = getBicyclicDiagonalRelation(
      RankedTensorType::get({n, p}, mlir::IndexType::get(&context)),
      /*contractionDim=*/0, /*stride=*/m, numSlots);
  auto packedB = evaluateLayoutOnMatrix(layoutB, matrixB);

  LiteralValue packedAValue = packedA[0];
  LiteralValue packedBValue = packedB;

  auto dag = implementRotateAndReduce(
      packedAValue, std::optional<LiteralValue>(packedBValue), m, n,
      DagType::intTensor(32, {numSlots}), /*zeroDiagonals=*/{}, "arith.addi");

  LiteralValue result = evalKernel(dag)[0];
  auto resultVec = std::get<std::vector<int>>(result.get());

  auto resultLayout = getBicyclicLayoutRelation(
      RankedTensorType::get({m, p}, mlir::IndexType::get(&context)), m * p);
  auto unpackedResult =
      unpackLayoutToMatrix<int>(resultLayout, {resultVec}, {m, p});

  std::vector<std::vector<int>> expected = {{386, 443, 471, 517, 565},
                                            {1461, 1692, 1814, 2000, 2214},
                                            {2879, 3350, 3612, 3986, 4440}};
  EXPECT_EQ(unpackedResult, expected);
}

TEST(KernelImplementationTest, BicyclicPtCtDiagonalMatmul) {
  MLIRContext context;
  std::vector<std::vector<int>> matrixA = {
      {1, 2, 3, 5}, {7, 11, 13, 17}, {19, 23, 29, 31}};
  std::vector<std::vector<int>> matrixB = {{2, 3, 5, 7, 11},
                                           {13, 17, 19, 23, 29},
                                           {31, 37, 41, 43, 47},
                                           {53, 59, 61, 67, 71}};
  int m = 3;
  int n = 4;
  int p = 5;
  int numSlots = 64;

  auto layoutA = getBicyclicDiagonalRelation(
      RankedTensorType::get({m, n}, mlir::IndexType::get(&context)),
      /*contractionDim=*/1, /*stride=*/p, numSlots);
  auto packedA = evaluateLayoutOnMatrix(layoutA, matrixA);

  auto layoutB = getBicyclicLayoutRelation(
      RankedTensorType::get({n, p}, mlir::IndexType::get(&context)), numSlots);
  auto packedB = evaluateLayoutOnMatrix(layoutB, matrixB);

  LiteralValue packedAValue = packedA;
  LiteralValue packedBValue = packedB[0];

  auto dag = implementRotateAndReduce(
      packedBValue, std::optional<LiteralValue>(packedAValue), p, n,
      DagType::intTensor(32, {numSlots}), /*zeroDiagonals=*/{}, "arith.addi");

  LiteralValue result = evalKernel(dag)[0];
  auto resultVec = std::get<std::vector<int>>(result.get());

  auto resultLayout = getBicyclicLayoutRelation(
      RankedTensorType::get({m, p}, mlir::IndexType::get(&context)), m * p);
  auto unpackedResult =
      unpackLayoutToMatrix<int>(resultLayout, {resultVec}, {m, p});

  std::vector<std::vector<int>> expected = {{386, 443, 471, 517, 565},
                                            {1461, 1692, 1814, 2000, 2214},
                                            {2879, 3350, 3612, 3986, 4440}};
  EXPECT_EQ(unpackedResult, expected);
}

TEST(KernelImplementationTest, BicyclicMatmulRotationCount) {
  MLIRContext context;

  // All triples (x, x+1, x+2) where x is odd are pairwise coprime.
  int m = 123;
  int n = 124;
  int p = 125;
  int numSlots = m * n * p;

  SymbolicValue packedAValue({m, n}, true);
  SymbolicValue packedBValue({n, p}, true);
  auto dag = implementBicyclicMatmul(packedAValue, packedBValue, m, n, p,
                                     DagType::intTensor(32, {numSlots}));

  RotationCountVisitor rotationCounter;
  int64_t rotationCount = rotationCounter.process(dag);

  // 124 + 2*math.sqrt(124) = 146
  EXPECT_EQ(rotationCount, 146);
}

TEST(KernelImplementationTest, BicyclicDiagonalMatmulRotationCount) {
  MLIRContext context;

  int m = 123;
  int n = 124;
  int p = 125;
  int numSlots = 2 * m * n * p;

  SymbolicValue secretVal({m, n}, true);
  SymbolicValue plainVal({n, p}, false);

  int period = m;
  int steps = n;
  auto dag = implementRotateAndReduce(
      secretVal, std::optional<SymbolicValue>(plainVal), period, steps,
      DagType::intTensor(32, {numSlots}));

  RotationCountVisitor rotationCounter;
  int64_t rotationCount = rotationCounter.process(dag);

  // 2*math.sqrt(124) = 22
  EXPECT_EQ(rotationCount, 22);
}

TEST(KernelImplementationTest, TricyclicBatchMatmul) {
  MLIRContext context;

  // Small example: h=2, m=3, n=5, p=2
  int64_t h = 2;
  int64_t m = 3;
  int64_t n = 5;
  int64_t p = 2;
  int64_t numSlots = h * m * n;

  // Build batch tensors A (h x m x n) and B (h x n x p) with deterministic
  // values
  std::vector<std::vector<std::vector<int>>> A(
      h, std::vector<std::vector<int>>(m, std::vector<int>(n, 0)));
  std::vector<std::vector<std::vector<int>>> B(
      h, std::vector<std::vector<int>>(n, std::vector<int>(p, 0)));

  int val = 1;
  for (int ih = 0; ih < h; ++ih)
    for (int im = 0; im < m; ++im)
      for (int in = 0; in < n; ++in) A[ih][im][in] = val++;
  val = 101;
  for (int ih = 0; ih < h; ++ih)
    for (int in = 0; in < n; ++in)
      for (int ip = 0; ip < p; ++ip) B[ih][in][ip] = val++;

  // Pack using the tricyclic layout relation.
  RankedTensorType typeA =
      RankedTensorType::get({h, m, n}, mlir::IndexType::get(&context));
  RankedTensorType typeB =
      RankedTensorType::get({h, n, p}, mlir::IndexType::get(&context));

  auto layoutA = getTricyclicLayoutRelation(typeA, numSlots);
  auto packedA = evaluateLayoutOnTensor(layoutA, A);

  auto layoutB = getTricyclicLayoutRelation(typeB, numSlots);
  auto packedB = evaluateLayoutOnTensor(layoutB, B);

  // Use ciphertext 0's slot vectors as literal inputs.
  LiteralValue packedAValue = packedA[0];
  LiteralValue packedBValue = packedB[0];

  // Generate kernel DAG and evaluate
  auto dag =
      implementTricyclicBatchMatmul(packedAValue, packedBValue, h, m, n, p,
                                    DagType::intTensor(32, {numSlots}));
  LiteralValue result = evalKernel(dag)[0];
  auto resultVec = std::get<std::vector<int>>(result.get());

  // Compute expected packed φ(Z) via evaluateLayoutOnTensor for the cleartext
  // result. First compute plain CPU batched matmul result (h x m x p).
  std::vector<std::vector<std::vector<int>>> expectedTensor(
      h, std::vector<std::vector<int>>(m, std::vector<int>(p, 0)));
  for (int ih = 0; ih < h; ++ih) {
    for (int im = 0; im < m; ++im) {
      for (int ip = 0; ip < p; ++ip) {
        int sum = 0;
        for (int ic = 0; ic < n; ++ic) {
          sum += A[ih][im][ic] * B[ih][ic][ip];
        }
        expectedTensor[ih][im][ip] = sum;
      }
    }
  }

  RankedTensorType resultType =
      RankedTensorType::get({h, m, p}, mlir::IndexType::get(&context));
  auto resultLayout = getTricyclicLayoutRelation(resultType, numSlots);
  auto expectedPacked = evaluateLayoutOnTensor(resultLayout, expectedTensor);

  // expectedPacked[0] is the expected slot vector for ciphertext 0.
  std::vector<int> expVec = expectedPacked[0];

  // Final check: compare expected packed φ(Z) vector with kernel output.
  EXPECT_EQ(expVec, resultVec);
}

TEST(KernelImplementationTest, TricyclicBatchMatmulRotationCount) {
  MLIRContext context;

  int h = 3;
  int m = 123;
  int n = 124;
  int p = 125;
  int numSlots = h * m * n * p;

  SymbolicValue packedAValue({h, m, n}, true);
  SymbolicValue packedBValue({h, n, p}, true);
  auto dag =
      implementTricyclicBatchMatmul(packedAValue, packedBValue, h, m, n, p,
                                    DagType::intTensor(32, {numSlots}));

  RotationCountVisitor rotationCounter;
  int64_t rotationCount = rotationCounter.process(dag);

  EXPECT_EQ(rotationCount, 146);
}

INSTANTIATE_TEST_SUITE_P(WithAndWithoutRolledSuite, KernelImplementationTest,
                         testing::Combine(testing::Values(false, true),
                                          testing::Values(false, true)));

TEST(KernelImplementationTest, TestDot) {
  std::vector<int> lhs = {1, 2, 3, 4};
  std::vector<int> rhs = {5, 6, 7, 8};
  std::vector<int> expected = {70, 70, 70, 70};

  LiteralValue lhsInput = lhs;
  LiteralValue rhsInput = rhs;

  auto dag = implementDot(lhsInput, rhsInput, 4, DagType::intTensor(32, {4}));
  LiteralValue result = evalKernel(dag)[0];
  auto actual = std::get<std::vector<int>>(result.get());

  EXPECT_EQ(actual, expected);
}

TEST(KernelImplementationTest, TestDotRotationCount) {
  SymbolicValue lhsVector({4}, true);
  SymbolicValue rhsVector({4}, true);
  auto dag = implementDot(lhsVector, rhsVector, 4, DagType::intTensor(32, {4}));

  RotationCountVisitor rotationCounter;
  int64_t rotationCount = rotationCounter.process(dag);

  EXPECT_EQ(rotationCount, 2);
}

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
