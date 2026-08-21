#ifndef LIB_UTILS_LAYOUT_CONVOLUTION_H_
#define LIB_UTILS_LAYOUT_CONVOLUTION_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "llvm/include/llvm/ADT/DenseSet.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {

// A gapped (pixel-shuffled) convolution layout folds a block of output channels
// into each spatial block: `gap * gap` channels for a 2-D conv, `gap` channels
// for a 1-D conv. That shuffle is a bijection only when the channel count is a
// multiple of the block, so a layout reserves the next multiple and leaves the
// extra channels empty. The expanded filter matrix has zero rows for them, and
// the result layout maps nothing into their slots.
int64_t getPaddedConvChannels(int64_t outputChannels, int64_t channelsPerBlock);

// Everything that fixes the expanded Toeplitz matrix a convolution lowers to,
// and the layout that matrix carries.
//
// A convolution becomes a plaintext-ciphertext matvec over that matrix. The
// choices below are made when layouts are assigned, but they are needed again
// when the kernel is materialized, so they travel together as one value. See
// tensor_ext::ConvPackingAttr for the form they take on an op.
struct ConvPacking {
  // The data operand the matrix is built against. It already accounts for
  // channel padding, and for a `tensor.pad` folded into `padding` below.
  RankedTensorType matrixDataType;
  // Symmetric spatial zero padding folded out of a `tensor.pad` and into the
  // convolution's own `padding` parameter.
  int64_t padding = 0;
  // Whether the matrix rows are pixel-shuffled. A shuffled matrix reserves
  // whole channel blocks, so this changes the matrix shape.
  bool interchangeRows = true;
  // The width the filter's diagonal layout was built at, when the data
  // operand's slot packing was absorbed into the matrix column space. The
  // columns are ciphertext slots then. Zero means nothing was absorbed.
  int64_t absorbedMatrixWidth = 0;

  // The expanded Toeplitz type this packing produces for `filterType`. Both
  // the layout assignment and the kernel materialization go through here, so
  // that one definition sizes the matrix.
  RankedTensorType expanded1dFilterType(RankedTensorType filterType,
                                        int64_t stride) const;
  RankedTensorType expanded2dFilterType(RankedTensorType filterType,
                                        ArrayRef<int64_t> strides) const;

  // The shape the filter's diagonal layout was built against. It is the
  // expanded Toeplitz shape, except that an absorbed packing makes the columns
  // ciphertext slots and so widens the layout.
  std::vector<int64_t> layoutMatrixShape(
      RankedTensorType expandedFilterType) const;
};

// The packing a 1-D convolution gets before any fold or absorption. Its rows
// always interchange.
ConvPacking defaultConv1dPacking(RankedTensorType dataType);

// The packing a 2-D convolution gets before any fold or absorption. Its rows
// interchange only when the convolution strides, because an unstrided one
// packs its rows and outputs densely with no channel gap.
ConvPacking defaultConv2dPacking(RankedTensorType dataType,
                                 ArrayRef<int64_t> strides);

// Removes `padding` entries from both ends of every spatial dim of a conv data
// operand: the width dim of a rank-3 (N, C, W) operand, or the height and
// width dims of a rank-4 (N, C, H, W) one. A conv's `padding` parameter is a
// single symmetric value shared by every spatial dim, which is why one
// `padding` covers them all.
//
// Returns nullopt when `padding` does not fit the operand, so that a caller
// cannot build a matrix against a degenerate shape.
std::optional<RankedTensorType> foldConvSpatialPadding(
    RankedTensorType dataType, int64_t padding);

// Returns an IntegerRelation that expands a 2-D filter matrix used in a
// convolution into a 2-D matrix such that the convolution is
// equivalent a matrix product with the flattened input vector. Each row
// corresponds to one filter multiplication. This does not include diagonalizing
// the matrix, the returned relation only expands the filter to the data matrix.
presburger::IntegerRelation get2dConvFilterRelation(RankedTensorType filterType,
                                                    RankedTensorType dataType,
                                                    ArrayRef<int64_t> strides,
                                                    int64_t padding);

// Returns an IntegerRelation that expands a 1-D filter used in a
// convolution into a 2-D matrix such that the convolution is
// equivalent a matrix product with the input vector. Each row
// corresponds to one filter multiplication. This does not include diagonalizing
// the matrix, the returned relation only expands the filter to the data matrix.
presburger::IntegerRelation get1dConvFilterRelation(RankedTensorType filterType,
                                                    RankedTensorType dataType,
                                                    int64_t stride,
                                                    int64_t padding);

RankedTensorType get2dConvFilterExpandedType(RankedTensorType filterType,
                                             RankedTensorType dataType,
                                             int64_t padding,
                                             ArrayRef<int64_t> strides);

RankedTensorType get1dConvFilterExpandedType(RankedTensorType filterType,
                                             RankedTensorType dataType,
                                             int64_t stride, int64_t padding);

// Returns an IntegerRelation that expands a filter matrix used in a
// convolution into a 2-D matrix such that the convolution is
// equivalent a matrix product with the flattened input vector. Each row
// corresponds to one filter multiplication.
FailureOr<presburger::IntegerRelation> getConvFilterDiagonalizedRelation(
    RankedTensorType filterType, RankedTensorType dataType, int64_t padding,
    int64_t minSlotCount);

// Returns an IntegerRelation that expands a multichannel filter used
// in a 2-D convolution into a 2-D Toeplitz matrix such that the convolution is
// equivalent a matrix product with the flattened multichannel input vector.
// Each row corresponds to one filter multiplication. This does not include
// diagonalizing the matrix, this simply returns the expanded data matrix. The
// filter type is assumed to be 4-D with dimensions (f, c, h, w) and the data
// type is assumed to be 3-D within a 4-D tensor of dimensions (1, c, h, w).
// Reads only the operand and its padding from `packing`; the rows are not
// interchanged here.
presburger::IntegerRelation get2dConvChwFchwFilterRelation(
    RankedTensorType filterType, const ConvPacking& packing,
    ArrayRef<int64_t> strides);

// Returns an IntegerRelation that expands a multichannel filter used
// in a 1-D convolution into a 2-D Toeplitz matrix such that the convolution is
// equivalent a matrix product with the flattened multichannel input vector.
// Each row corresponds to one filter multiplication. This does not include
// diagonalizing the matrix, this simply returns the expanded data matrix. The
// filter type is assumed to be 3-D with dimensions (f, c, w) and the data
// type is assumed to be 2-D with dimensions (1, c, w).
// Reads only the operand and its padding from `packing`; the rows are not
// interchanged here.
presburger::IntegerRelation get1dConvCwFcwFilterRelation(
    RankedTensorType filterType, const ConvPacking& packing, int64_t stride);

// `interchangeRows` must match the flag the filter layout was built with: an
// interchanged (pixel-shuffled) layout reserves whole channel blocks, so its
// matrix has extra zero rows when the channel count is not a multiple of the
// block. The Halevi-Shoup kernel is sized from this type, so it has to agree
// with the layout relation.
//
// The flag travels on `packing`, beside the operand it was decided against, so
// that one value cannot size a matrix one way and lay out its rows the other.
RankedTensorType get1dConvCwFcwFilterExpandedType(RankedTensorType filterType,
                                                  const ConvPacking& packing,
                                                  int64_t stride);

RankedTensorType get2dConvChwFchwFilterExpandedType(RankedTensorType filterType,
                                                    const ConvPacking& packing,
                                                    ArrayRef<int64_t> strides);

// Returns an IntegerRelation that represents a diagonalized 2-D Toeplitz matrix
// that is used to compute a 1-D multichannel convolution filter such that the
// convolution is equivalent a matrix product with the flattened multichannel
// input vector. Each row corresponds to one filter multiplication. The filter
// type is assumed to be 3-D with dimensions (f, c, w) and the data type is
// assumed to be 3-D with dimensions (1, c, w).
// `dataSlotPermutation`, when non-null, maps the flattened data index [j] to
// the [ct, slot] the element actually occupies. It re-indexes the matrix's
// column space by that packing, so the diagonal kernel reads the data where it
// already sits instead of converting the ciphertext. It must give each column
// at most one slot and no two columns the same slot; see
// getDiagonalColumnRepresentative. A column with no slot is dropped, which is
// correct exactly when that element is zero.
FailureOr<presburger::IntegerRelation> get1dConvCwFcwFilterDiagonalizedRelation(
    RankedTensorType filterType, const ConvPacking& packing, int64_t stride,
    int64_t minSlotCount,
    const presburger::IntegerRelation* dataSlotPermutation = nullptr);

// Flattens a conv data layout `[n, c, spatial...] -> [ct, slot]` into the
// column-space permutation `[j] -> [ct, slot]`, where j is the row-major index
// of the matrix operand: j = c * W + w for a 3-D (1, C, W) operand and
// j = c * H * W + h * W + w for a 4-D (1, C, H, W) one. Accepted as the
// `dataSlotPermutation` of the diagonalized relation of either rank.
//
// `packing.matrixDataType` is the operand the Toeplitz matrix is built against,
// so it fixes W and therefore the column space. `packing.padding` is nonzero
// when a `tensor.pad` folded into the conv: the matrix is then built against
// the unpadded operand while `dataLayout` still indexes the padded value, so
// column j must read the slot of padded index (c, w + padding).
//
// Fails if the layout does not pack the data into ciphertext zero, and, when
// the padding is nonzero, if the shifted window leaves any column without a
// slot. Every column is real data in that case, so dropping one would drop
// data.
// Reads only the operand and its padding from `packing`.
FailureOr<presburger::IntegerRelation> getConvDataColumnPermutation(
    const ConvPacking& packing, const presburger::IntegerRelation& dataLayout);

// The columns j = c * W + w that `columnPermutation` gives a slot to read. The
// diagonal kernel drops any column outside this set from the plaintext matrix,
// so a caller that absorbs a packing must check that those elements are zero.
llvm::DenseSet<int64_t> getMappedConvMatrixColumns(
    const presburger::IntegerRelation& columnPermutation);

// Returns a sequence of IntegerRelations that represents the layout mapping as
// a series of simple steps (Toeplitz expansion, row interchange, flattening,
// diagonalization). This is preferred for compilation performance to avoid ISL
// hangs when generating loops.
FailureOr<std::vector<presburger::IntegerRelation>>
// `dataSlotPermutation` carries the same meaning as it does for the 1-D
// diagonalized relation: when non-null it re-indexes the matrix's column space
// by the slot each data element occupies.
get2dConvChwFchwFilterAsSequence(
    RankedTensorType filterType, const ConvPacking& packing,
    ArrayRef<int64_t> strides, int64_t minSlotCount,
    const presburger::IntegerRelation* dataSlotPermutation = nullptr);

// Returns an IntegerRelation for a row-interchange map that optimizes the
// diagonal structure of a convolution's Toeplitz matrix.
//
// It maps flattened indices from a channel-last (H, W, C*g^2) tensor to a
// (gH, gW, C) tensor. This rearrangement interleaves sub-pixels
// from the channel dimension into g x g spatial blocks, effectively performing
// a depth-to-space (pixel-shuffle) operation.
// See Orion's implementation of multiplex:
// https://github.com/baahl-nyu/orion/blob/0f7df1717be44e21caeab42f8a9da81c997fe7e8/orion/core/packing.py#L159
// This computes the flattened input to flattened output map, e.g.
// input = torch.arange(n * c * h * w).reshape(n, c, h, w)
// result = multiplex(input, gap)
// flattened_result = result.squeeze(0).flatten()
presburger::IntegerRelation get2dConvRowInterchangeRelation(int64_t c,
                                                            int64_t h,
                                                            int64_t w,
                                                            int64_t g);

// Returns an IntegerRelation for a row-interchange map that optimizes the
// diagonal structure of a convolution's Toeplitz matrix.
presburger::IntegerRelation get1dConvRowInterchangeRelation(int64_t c,
                                                            int64_t w,
                                                            int64_t g);

bool isRelationConvFilterDiagonalized(
    RankedTensorType filterType, RankedTensorType dataType, int64_t padding,
    int64_t minSlotCount, const presburger::IntegerRelation& relation);

// Returns an IntegerRelation that corresponds to the output layout of a 1-D
// multi-channel convolution. This includes the row interchange from pixel
// shuffling. The result is a relation mapping to (ct, slot) of the output.
presburger::IntegerRelation get1dConvResultRelation(RankedTensorType outputType,
                                                    int64_t stride,
                                                    int64_t padding,
                                                    int64_t minSlotCount,
                                                    bool interchangeRows);

// Returns an IntegerRelation that corresponds to the output layout of a 2-D
// multi-channel convolution. This includes the row interchange from pixel
// shuffling. The result is a relation mapping to (ct, slot) of the output.
//
// Set `interchangeRows` when the caller composes this with
// get2dConvRowInterchangeLayoutRelation: a shuffled result reserves whole
// channel blocks, and both relations must use that same larger extent as the
// replication period.
//
// This parameter used to default to false while the rest of the family
// defaulted to true, so a caller that composed but omitted it got a result
// layout narrower than the matrix it belonged to. Nothing defaults now.
presburger::IntegerRelation get2dConvResultRelation(RankedTensorType outputType,
                                                    ArrayRef<int64_t> strides,
                                                    int64_t padding,
                                                    int64_t minSlotCount,
                                                    bool interchangeRows);

presburger::IntegerRelation get2dConvRowInterchangeLayoutRelation(
    RankedTensorType outputType, ArrayRef<int64_t> strides,
    int64_t minSlotCount);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_CONVOLUTION_H_
