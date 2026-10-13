package depthwiseconv2dstrided

import (
	"math"
	"testing"

	"tests/Examples/lattigo/ckks/depthwise_conv2d_strided/depthwiseconv2dstrided_utils"
)

func TestDepthwiseConv2DStrided(t *testing.T) {
	evaluator, params, ecd, enc, dec := Depthwise_conv2d_strided__configure()

	// 1x4x8x8 input, row-major over (c, h, w).
	const c, h, w = 4, 8, 8
	arg0 := make([]float32, c*h*w)
	for i := range arg0 {
		arg0[i] = float32(i%11)/10 - 0.5
	}

	// 3x3 depthwise conv with stride 2, padding 1 and bias, then a 1x1 conv.
	dwFilter := [c][3][3]float64{
		{{1.0, 0.0, -1.0}, {0.5, 1.0, 0.5}, {-1.0, 0.0, 1.0}},
		{{0.0, 0.5, 0.0}, {0.5, -1.0, 0.5}, {0.0, 0.5, 0.0}},
		{{-0.5, 1.0, -0.5}, {1.0, 0.0, -1.0}, {0.5, -1.0, 0.5}},
		{{0.25, 0.25, 0.25}, {0.25, 0.5, 0.25}, {0.25, 0.25, 0.25}},
	}
	dwBias := [c]float64{0.1, -0.1, 0.2, 0.0}
	pwFilter := [c][c]float64{
		{0.5, -1.0, 1.0, 0.25},
		{-0.5, 0.5, 1.0, -1.0},
		{1.0, 0.25, -0.5, 0.5},
		{0.0, 1.0, 0.5, -0.25},
	}
	pwBias := [c]float64{0.0, 0.1, -0.1, 0.2}

	const oh, ow = 4, 4
	var dw [c][oh][ow]float64
	for ch := 0; ch < c; ch++ {
		for y := 0; y < oh; y++ {
			for x := 0; x < ow; x++ {
				sum := dwBias[ch]
				for ky := 0; ky < 3; ky++ {
					for kx := 0; kx < 3; kx++ {
						iy, ix := 2*y+ky-1, 2*x+kx-1
						if iy < 0 || iy >= h || ix < 0 || ix >= w {
							continue
						}
						sum += dwFilter[ch][ky][kx] * float64(arg0[(ch*h+iy)*w+ix])
					}
				}
				dw[ch][y][x] = sum
			}
		}
	}
	expected := make([]float64, c*oh*ow)
	for o := 0; o < c; o++ {
		for y := 0; y < oh; y++ {
			for x := 0; x < ow; x++ {
				sum := pwBias[o]
				for ch := 0; ch < c; ch++ {
					sum += pwFilter[o][ch] * dw[ch][y][x]
				}
				expected[(o*oh+y)*ow+x] = sum
			}
		}
	}

	ct0 := Depthwise_conv2d_strided__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	linearTransforms, filterPlains := depthwiseconv2dstrided_utils.Depthwise_conv2d_strided__preprocessing(params, ecd)
	resultCt := Depthwise_conv2d_strided__preprocessed(evaluator, params, ecd, ct0, linearTransforms, filterPlains)
	result := Depthwise_conv2d_strided__decrypt__result0(evaluator, params, ecd, dec, resultCt)
	errorThreshold := float64(0.01)
	for i := range expected {
		if math.Abs(float64(result[i])-expected[i]) > errorThreshold {
			t.Errorf("Decryption error at index %d: got %.4f, expected %.4f", i, result[i], expected[i])
		}
	}
}
