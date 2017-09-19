package util

import (
	"math"
	"math/rand"
	"time"
)

type GaussianDistribution struct {
	mean float64
	val  float64
}

func NewGaussianDistribution(mean, val float64) *GaussianDistribution {
	return &GaussianDistribution{mean, val}
}

func (g GaussianDistribution) Random() float64 {
	rand.Seed(time.Now().UnixNano())
	var r float64 = 0.0
	for r == 0.0 {
		r = rand.Float64()
	}
	var c float64 = math.Sqrt(-2.0 * math.Log(r))

	if rand.Float64() < 0.5 {
		return c*math.Sin(2.0*math.Pi*rand.Float64())*g.val + g.mean
	}
	return c*math.Cos(2.0*math.Pi*rand.Float64())*g.val + g.mean
}
