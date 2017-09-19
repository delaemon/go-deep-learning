package activation

import "math"

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Pow(math.E, x))
}

func step(x float64) int {
	if x >= 0 {
		return 1
	}
	return -1
}
