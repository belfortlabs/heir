package conv2dnchwgappedpadded

import (
	"math"
	"testing"
)

// The second conv in this chain takes a data operand that is both gapped (the
// first conv has stride 2, so its result is pixel-shuffled) and padded.
// LayoutPropagation folds that pad into the conv's own padding parameter and
// absorbs the gapped packing into the plaintext diagonal filter, so this checks
// that the absorbed matrix computes the same numbers a plain convolution does.
func TestConv2DGappedPadded(t *testing.T) {
	evaluator, params, ecd, enc, dec := Conv2d_nchw_gapped_padded__configure()

	arg0 := make([]float32, 32)
	for i := 0; i < 32; i++ {
		arg0[i] = float32(i) * 0.1
	}

	expected := []float32{
		6.06719, 5.14219, 1.05937, -0.303125,
		-0.303125, -4.29531, -1.49219, -2.40469,
	}

	ct0 := Conv2d_nchw_gapped_padded__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := Conv2d_nchw_gapped_padded(evaluator, params, ecd, ct0)
	result := Conv2d_nchw_gapped_padded__decrypt__result0(evaluator, params, ecd, dec, resultCt)
	errorThreshold := float64(0.05)
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.4f != %.4f", i, result[i], expected[i])
		}
	}
}
