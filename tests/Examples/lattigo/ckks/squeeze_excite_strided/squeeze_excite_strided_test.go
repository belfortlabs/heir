package squeezeexcitestrided

import (
	"math"
	"testing"
)

func TestSqueeze_excite_strided(t *testing.T) {
	evaluator, params, ecd, enc, dec := Squeeze_excite_strided__configure()

	// 1x1x8x8 input, row-major.
	x := []float32{-0.60, 0.10, -0.50, 0.20, -0.40, 0.30, -0.30, 0.40, -0.20, 0.50, -0.10, 0.60, 0.00, -0.60, 0.10, -0.50, 0.20, -0.40, 0.30, -0.30, 0.40, -0.20, 0.50, -0.10, 0.60, 0.00, -0.60, 0.10, -0.50, 0.20, -0.40, 0.30, -0.30, 0.40, -0.20, 0.50, -0.10, 0.60, 0.00, -0.60, 0.10, -0.50, 0.20, -0.40, 0.30, -0.30, 0.40, -0.20, 0.50, -0.10, 0.60, 0.00, -0.60, 0.10, -0.50, 0.20, -0.40, 0.30, -0.30, 0.40, -0.20, 0.50, -0.10, 0.60}
	// Per-channel scale, 1x4x1x1.
	s := []float32{0.5, -1.0, 1.5, 2.0}
	// expected[c][h][w] = s[c] * filter[c] * x[2h][2w] with filter = [1, 2, -1, 0.5]
	expected := []float32{-0.300, -0.250, -0.200, -0.150, 0.100, 0.150, 0.200, 0.250, -0.150, -0.100, -0.050, 0.000, 0.250, 0.300, -0.300, -0.250, 1.200, 1.000, 0.800, 0.600, -0.400, -0.600, -0.800, -1.000, 0.600, 0.400, 0.200, 0.000, -1.000, -1.200, 1.200, 1.000, 0.900, 0.750, 0.600, 0.450, -0.300, -0.450, -0.600, -0.750, 0.450, 0.300, 0.150, 0.000, -0.750, -0.900, 0.900, 0.750, -0.600, -0.500, -0.400, -0.300, 0.200, 0.300, 0.400, 0.500, -0.300, -0.200, -0.100, 0.000, 0.500, 0.600, -0.600, -0.500}

	ctX := Squeeze_excite_strided__encrypt__arg0(evaluator, params, ecd, enc, x)
	ctS := Squeeze_excite_strided__encrypt__arg1(evaluator, params, ecd, enc, s)
	resultCt := Squeeze_excite_strided(evaluator, params, ecd, ctX, ctS)
	result := Squeeze_excite_strided__decrypt__result0(evaluator, params, ecd, dec, resultCt)
	errorThreshold := float64(0.01)
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.4f != %.4f", i, result[i], expected[i])
		}
	}
}
