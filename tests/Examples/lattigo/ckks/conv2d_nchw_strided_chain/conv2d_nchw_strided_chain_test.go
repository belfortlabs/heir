package conv2dnchwstridedchain

import (
	"math"
	"testing"

	"tests/Examples/lattigo/ckks/conv2d_nchw_strided_chain/conv2dnchwstridedchain_utils"
)

// conv2dNchwFchw computes an unpadded 2D convolution over a row-major NCHW
// input with an FCHW filter.
func conv2dNchwFchw(input, filter []float64, cin, hin, win, cout, k, stride int) []float64 {
	hout := (hin-k)/stride + 1
	wout := (win-k)/stride + 1
	output := make([]float64, cout*hout*wout)
	for f := 0; f < cout; f++ {
		for ho := 0; ho < hout; ho++ {
			for wo := 0; wo < wout; wo++ {
				var sum float64
				for c := 0; c < cin; c++ {
					for kh := 0; kh < k; kh++ {
						for kw := 0; kw < k; kw++ {
							in := input[c*hin*win+(ho*stride+kh)*win+wo*stride+kw]
							sum += in * filter[((f*cin+c)*k+kh)*k+kw]
						}
					}
				}
				output[(f*hout+ho)*wout+wo] = sum
			}
		}
	}
	return output
}

// The MLIR filter constants were generated from these formulas.
func makeFilter(cout, cin int, weight func(f, c, kh, kw int) float64) []float64 {
	filter := make([]float64, cout*cin*4)
	for f := 0; f < cout; f++ {
		for c := 0; c < cin; c++ {
			for kh := 0; kh < 2; kh++ {
				for kw := 0; kw < 2; kw++ {
					filter[((f*cin+c)*2+kh)*2+kw] = weight(f, c, kh, kw)
				}
			}
		}
	}
	return filter
}

func TestConv2dStridedChain(t *testing.T) {
	evaluator, params, ecd, enc, dec := Conv2d_strided_chain__configure()

	arg0 := make([]float32, 64)
	input := make([]float64, 64)
	for i := range arg0 {
		input[i] = float64((i*5)%11-5) / 5
		arg0[i] = float32(input[i])
	}
	filter1 := makeFilter(4, 1, func(f, c, kh, kw int) float64 {
		return float64((f*3+c*5+kh*2+kw)%7-3) / 4
	})
	filter2 := makeFilter(5, 4, func(f, c, kh, kw int) float64 {
		return float64((f*5+c*3+kh*2+kw*3)%9-4) / 8
	})
	hidden := conv2dNchwFchw(input, filter1, 1, 8, 8, 4, 2, 2)
	expected := conv2dNchwFchw(hidden, filter2, 4, 4, 4, 5, 2, 2)

	ct0 := Conv2d_strided_chain__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	linearTransforms, filterPlains := conv2dnchwstridedchain_utils.Conv2d_strided_chain__preprocessing(params, ecd)
	resultCt := Conv2d_strided_chain__preprocessed(evaluator, params, ecd, ct0, linearTransforms, filterPlains)
	result := Conv2d_strided_chain__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	if len(result) != len(expected) {
		t.Fatalf("got %d outputs, expected %d", len(result), len(expected))
	}
	for i := range expected {
		if math.Abs(float64(result[i])-expected[i]) > 0.01 {
			t.Errorf("index %d: got %.4f, expected %.4f", i, result[i], expected[i])
		}
	}
}
