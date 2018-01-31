package util

import "math/rand"

func Uniform(min, max float64) float64 {
	return rand.Float64()*(max-min) + min
}

func Binomial(n int, p float64) int {
	if p < 0 || p > 1 {
		return 0
	}
	c := 0
	for i := 0; i < n; i++ {
		r := rand.Float64()
		if r < p {
			c++
		}
	}
	return c
}
