package util

import "math"

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Pow(math.E, x))
}

func Desigmoid(x float64) float64 {
	return x * (1.0 - x)
}

func Tanh(x float64) float64 {
	return math.Tanh(x)
}

func Detanh(x float64) float64 {
	return 1.0 - x*x
}

func ReLU(x float64) float64 {
	if x > 0 {
		return x
	} else {
		return 0.0
	}
}

func DeReLU(x float64) float64 {
	if x > 0 {
		return 1.0
	} else {
		return 0.0
	}
}

func Step(x float64) int {
	if x >= 0 {
		return 1
	}
	return -1
}

func Softmax(x []float64, n int) []float64 {
	y := make([]float64, n)
	var max float64 = 0
	var sum float64 = 0

	for i := 0; i < n; i++ {
		if max < x[i] {
			max = x[i]
		}
	}

	for i := 0; i < n; i++ {
		y[i] = math.Exp(x[i] - max)
		sum += y[i]
	}

	for i := 0; i < n; i++ {
		y[i] /= sum
	}

	return y
}
