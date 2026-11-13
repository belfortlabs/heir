package squeezeexcitemultict

import (
	"math"
	"testing"
)

func TestSqueeze_excite_multi_ct(t *testing.T) {
	evaluator, params, ecd, enc, dec := Squeeze_excite_multi_ct__configure()

	// 1x8x16x16 input, row-major, and a 1x8x1x1 per-channel scale.
	x := make([]float32, 8*16*16)
	for i := range x {
		x[i] = float32(i%13-6) / 10
	}
	s := []float32{0.5, -1.0, 1.5, 2.0, -0.5, 1.0, -1.5, 0.25}
	expected := make([]float32, len(x))
	for i := range x {
		expected[i] = s[i/256] * x[i]
	}

	ctX := Squeeze_excite_multi_ct__encrypt__arg0(evaluator, params, ecd, enc, x)
	ctS := Squeeze_excite_multi_ct__encrypt__arg1(evaluator, params, ecd, enc, s)
	resultCt := Squeeze_excite_multi_ct(evaluator, params, ecd, ctX, ctS)
	result := Squeeze_excite_multi_ct__decrypt__result0(evaluator, params, ecd, dec, resultCt)
	errorThreshold := float64(0.01)
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.4f != %.4f", i, result[i], expected[i])
		}
	}
}
