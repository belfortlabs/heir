package depthwiseconv2d

import (
	"math"
	"testing"

	"tests/Examples/lattigo/ckks/depthwise_conv2d/depthwiseconv2d_utils"
)

func TestDepthwiseConv2D(t *testing.T) {
	evaluator, params, ecd, enc, dec := Depthwise_conv2d__configure()

	// 1x4x8x8 input, row-major over (c, h, w).
	arg0 := make([]float32, 256)
	for i := range arg0 {
		arg0[i] = float32(i%11)/10 - 0.5
	}

	// 3x3 depthwise conv with padding 1 and bias, then a 1x1 conv to 2 channels.
	expected := []float32{
		0.0750, 0.2875, -1.2688, 0.4750, 1.1188, 0.5938, 0.0000, 1.1875,
		1.0062, 0.2937, 0.5625, 0.6250, -0.1375, -1.2437, 0.4000, 0.4687,
		0.2062, -0.2375, 1.4062, 0.8500, 0.2937, 0.5625, 0.6250, -0.4500,
		-1.2125, 0.4000, 0.9438, 0.3875, -0.2375, 1.4062, 0.8500, 0.4875,
		1.1500, 0.6250, -0.1375, -1.2437, 0.4000, 0.9438, 0.3875, 0.1188,
		1.1062, 0.8500, 0.2937, 0.5625, 0.6250, -0.1375, -1.2437, -0.1812,
		-0.2438, 0.3875, -0.2375, 1.4062, 0.8500, 0.2937, 0.5625, 0.7562,
		0.3562, -0.6250, 1.0500, 1.0750, 0.0000, -0.0437, 0.5312, 0.5812,
		0.2000, 0.1250, 0.9750, 1.5500, 0.7500, -0.6000, -0.8500, 0.4750,
		0.2250, -0.0750, -0.1000, 0.7000, -0.7000, 0.9250, 1.7250, 0.3500,
		0.3750, -0.5500, -0.0250, -0.6000, -0.0750, -0.1000, 0.7000, -0.1000,
		-0.0250, 1.7250, 1.1500, -0.2500, -0.5500, -0.0250, -0.6000, 0.5500,
		-0.1500, 0.7000, -0.7000, 0.9250, 1.7250, 1.1500, -0.2500, -0.7250,
		0.2750, -0.6000, -0.0750, -0.1000, 0.7000, -0.7000, 0.9250, 0.7500,
		0.7000, -0.2500, -0.5500, -0.0250, -0.6000, -0.0750, -0.1000, 0.8500,
		-1.0000, 0.3250, 1.1750, 1.2000, 0.1250, 0.4250, -0.1000, 0.0250,
	}

	ct0 := Depthwise_conv2d__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	linearTransforms, filterPlains := depthwiseconv2d_utils.Depthwise_conv2d__preprocessing(params, ecd)
	resultCt := Depthwise_conv2d__preprocessed(evaluator, params, ecd, ct0, linearTransforms, filterPlains)
	result := Depthwise_conv2d__decrypt__result0(evaluator, params, ecd, dec, resultCt)
	errorThreshold := float64(0.01)
	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: got %.4f, expected %.4f", i, result[i], expected[i])
		}
	}
}
