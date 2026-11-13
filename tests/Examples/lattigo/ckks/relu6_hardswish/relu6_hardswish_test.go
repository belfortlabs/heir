package relu6hardswish

import (
	"math"
	"testing"
)

func relu6(x float64) float64 {
	return math.Min(math.Max(x, 0), 6)
}

func TestRelu6Hardswish(t *testing.T) {
	evaluator, params, ecd, enc, dec := Relu6_hardswish__configure()

	const n = 16
	arg0 := make([]float32, n)
	expected := make([]float64, n)
	for i := 0; i < n; i++ {
		x := float64(i)*0.9 - 7.0
		arg0[i] = float32(x)
		expected[i] = relu6(x) + x*relu6(x+3)/6
	}

	ct0 := Relu6_hardswish__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := Relu6_hardswish(evaluator, params, ecd, ct0)
	result := Relu6_hardswish__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	const errorThreshold = 0.01
	worst := 0.0
	for i := range expected {
		diff := math.Abs(float64(result[i]) - expected[i])
		worst = math.Max(worst, diff)
		t.Logf("x=%.2f: got %.4f want %.4f", arg0[i], result[i], expected[i])
		if diff > errorThreshold {
			t.Errorf("index %d (x=%.2f): got %.4f want %.4f (err %.4f)",
				i, arg0[i], result[i], expected[i], diff)
		}
	}
	t.Logf("worst absolute error: %.6f", worst)
}
