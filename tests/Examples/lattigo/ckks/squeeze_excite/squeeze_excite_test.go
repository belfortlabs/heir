package squeezeexcite

import (
	"math"
	"testing"
)

func TestSqueeze_excite(t *testing.T) {
	evaluator, params, ecd, enc, dec := Squeeze_excite__configure()

	// 1x4x4x4 input, row-major.
	x := []float32{-0.60, 0.10, -0.50, 0.20, -0.40, 0.30, -0.30, 0.40, -0.20, 0.50, -0.10, 0.60, 0.00, -0.60, 0.10, -0.50, 0.20, -0.40, 0.30, -0.30, 0.40, -0.20, 0.50, -0.10, 0.60, 0.00, -0.60, 0.10, -0.50, 0.20, -0.40, 0.30, -0.30, 0.40, -0.20, 0.50, -0.10, 0.60, 0.00, -0.60, 0.10, -0.50, 0.20, -0.40, 0.30, -0.30, 0.40, -0.20, 0.50, -0.10, 0.60, 0.00, -0.60, 0.10, -0.50, 0.20, -0.40, 0.30, -0.30, 0.40, -0.20, 0.50, -0.10, 0.60}
	// Per-channel scale, 1x4x1x1.
	s := []float32{0.5, -1.0, 1.5, 2.0}
	// expected[c][h][w] = s[c] * x[c][h][w]
	expected := []float32{-0.30, 0.05, -0.25, 0.10, -0.20, 0.15, -0.15, 0.20, -0.10, 0.25, -0.05, 0.30, 0.00, -0.30, 0.05, -0.25, -0.20, 0.40, -0.30, 0.30, -0.40, 0.20, -0.50, 0.10, -0.60, 0.00, 0.60, -0.10, 0.50, -0.20, 0.40, -0.30, -0.45, 0.60, -0.30, 0.75, -0.15, 0.90, 0.00, -0.90, 0.15, -0.75, 0.30, -0.60, 0.45, -0.45, 0.60, -0.30, 1.00, -0.20, 1.20, 0.00, -1.20, 0.20, -1.00, 0.40, -0.80, 0.60, -0.60, 0.80, -0.40, 1.00, -0.20, 1.20}

	ctX := Squeeze_excite__encrypt__arg0(evaluator, params, ecd, enc, x)
	ctS := Squeeze_excite__encrypt__arg1(evaluator, params, ecd, enc, s)
	resultCt := Squeeze_excite(evaluator, params, ecd, ctX, ctS)
	result := Squeeze_excite__decrypt__result0(evaluator, params, ecd, dec, resultCt)
	errorThreshold := float64(0.01)
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.4f != %.4f", i, result[i], expected[i])
		}
	}
}
