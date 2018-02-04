package mln

import "github.com/delaemon/go-deep-learning/util"

type HiddenLayer struct {
	n_in         int
	n_out        int
	w            [MLP_N_out][MLP_N_in]float64
	b            [MLP_N_out]float64
	activation   func(x float64) float64
	deactivation func(x float64) float64
}

func NewHiddenLayer(activation string) *HiddenLayer {
	w := [MLP_N_out][MLP_N_in]float64{}
	var w_ float64 = 1.0 / MLP_N_in
	for j := 0; j < MLP_N_out; j++ {
		for i := 0; i < MLP_N_in; i++ {
			w[j][i] = util.Uniform(-w_, w_)
		}
	}

	var act func(x float64) float64
	var deact func(x float64) float64
	if activation == "sigmoid" {
		act = util.Sigmoid
		deact = util.Desigmoid
	} else if activation == "tanh" {
		act = util.Tanh
		deact = util.Detanh
	} else if activation == "ReLU" {
		act = util.ReLU
		deact = util.DeReLU
	} else {
		panic("activation function not supported")
	}

	return &HiddenLayer{
		w:            w,
		activation:   act,
		deactivation: deact,
	}
}

func (h *HiddenLayer) output(x [MLP_N_in]float64) [MLP_N_out]float64 {
	y := [MLP_N_out]float64{}
	for j := 0; j < MLP_N_out; j++ {
		preActivation_ := 0.0
		for i := 0; i < MLP_N_in; i++ {
			preActivation_ += h.w[j][i] * x[i]
		}
		preActivation_ += h.b[j]
		y[j] = h.activation(preActivation_)
	}
	return y
}

func (h *HiddenLayer) forward(x [MLP_N_in]float64) [MLP_N_out]float64 {
	return h.output(x)
}

func (h *HiddenLayer) backward(x [MLP_MinibatchSize][MLP_N_in]float64, z [MLP_MinibatchSize][MLP_N_in]float64, dy [MLP_MinibatchSize][MLP_N_out]float64, wprev [MLP_N_out][MLP_N_in]float64) [MLP_MinibatchSize][MLP_N_out]float64 {
	dz := [MLP_MinibatchSize][MLP_N_out]float64{}
	grad_w := [MLP_N_out][MLP_N_in]float64{}
	grad_b := [MLP_N_out]float64{}

	for n := 0; n < MLP_MinibatchSize; n++ {
		for j := 0; j < MLP_N_out; j++ {
			for k := 0; k < len(dy[0]); k++ {
				dz[n][j] += wprev[k][j] * dy[n][k]
			}
			dz[n][j] *= h.deactivation(z[n][j])

			for i := 0; i < MLP_N_in; i++ {
				grad_w[j][i] += dz[n][j] * x[n][i]
			}
			grad_b[j] += dz[n][j]
		}
	}

	for j := 0; j < MLP_N_out; j++ {
		for i := 0; i < MLP_N_in; i++ {
			h.w[j][i] -= MLP_LearningRate * grad_w[j][i] / MLP_MinibatchSize
		}
		h.b[j] -= MLP_LearningRate * grad_b[j] / MLP_MinibatchSize
	}
	return dz
}
