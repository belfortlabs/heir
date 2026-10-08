package conv2dnchwstridedpad

import (
	"math"
	"testing"
)

// A stride-2 conv, a zero pad and a stride-1 conv: the pad sits on the gapped
// result of the first conv.
func TestConv2DStridedPad(t *testing.T) {
	evaluator, params, ecd, enc, dec := Conv2d_nchw_strided_pad__configure()

	// 1x1x8x8 input, row-major over (h, w).
	arg0 := []float32{
		0.00, 0.40, 0.80, 1.20, 0.15, 0.55, 0.95, 1.35,
		0.30, 0.70, 1.10, 0.05, 0.45, 0.85, 1.25, 0.20,
		0.60, 1.00, 1.40, 0.35, 0.75, 1.15, 0.10, 0.50,
		0.90, 1.30, 0.25, 0.65, 1.05, 0.00, 0.40, 0.80,
		1.20, 0.15, 0.55, 0.95, 1.35, 0.30, 0.70, 1.10,
		0.05, 0.45, 0.85, 1.25, 0.20, 0.60, 1.00, 1.40,
		0.35, 0.75, 1.15, 0.10, 0.50, 0.90, 1.30, 0.25,
		0.65, 1.05, 0.00, 0.40, 0.80, 1.20, 0.15, 0.55,
	}

	expected := []float32{
		1.9625, 0.0000, 1.6875, -1.5250, -0.8375, -1.4375, 4.6875, -1.7750,
		-0.6500, 2.2500, -3.9500, 5.8625, -0.8375, -3.2875, 0.2625, -0.1250,
		-2.8750, 4.5750, -1.9125, 0.3250, 2.7375, -3.4125, 1.0125, -0.3375,
		0.6500, 1.5375, 0.1625, -3.4125, -1.7000, 3.8000, -0.6375, -0.9500,
		0.7875, -3.5375, 4.3625, -1.2000, -1.3125, 2.0500, -5.4750, 4.6625,
		-1.7375, -1.2375, -0.7875, -0.6875, 3.8750, -4.0500, 1.2750, -1.7750,
		-0.7375, -0.9625, -1.0500, 1.4000, 2.8250, -0.9250, 3.5375, -3.7750,
		-2.1250, 3.2375, 2.2625, -0.2125, -0.9875, 2.3500, -0.3125, 4.0250,
	}

	ct0 := Conv2d_nchw_strided_pad__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := Conv2d_nchw_strided_pad(evaluator, params, ecd, ct0)
	result := Conv2d_nchw_strided_pad__decrypt__result0(evaluator, params, ecd, dec, resultCt)
	errorThreshold := float64(0.05)
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.4f != %.4f", i, result[i], expected[i])
		}
	}
}
