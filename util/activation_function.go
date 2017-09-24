package util

import "math"

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Pow(math.E, x))
}

func Step(x float64) int {
	if x >= 0 {
		return 1
	}
	return -1
}
