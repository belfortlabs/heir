package levelzeroencryption

import (
	"math"
	"testing"
)

func TestLevelZeroEncryption(t *testing.T) {
	btEvaluator, evaluator, params, ecd, enc, dec := Dot_product__configure()

	// Vector of plaintext values
	arg0 := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	arg1 := []float32{0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}

	expected := float32(2.50)

	ct0 := Dot_product__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	ct1 := Dot_product__encrypt__arg1(evaluator, params, ecd, enc, arg1)

	if ct0[0].Level() != 0 || ct1[0].Level() != 0 {
		t.Fatalf("encryption helpers produced levels %d and %d, want 0 and 0",
			ct0[0].Level(), ct1[0].Level())
	}

	resultCt := Dot_product(btEvaluator, evaluator, params, ecd, ct0, ct1)

	result := Dot_product__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	// Bootstrapping is the dominant error source here, and it costs about a
	// decimal digit against the plain dot_product_8f test's 1e-4 tolerance.
	errorThreshold := float64(0.001)
	if math.Abs(float64(result-expected)) > errorThreshold {
		t.Errorf("Decryption error %.4f != %.4f", result, expected)
	}
}
