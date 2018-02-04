package mln

import "github.com/delaemon/go-deep-learning/util"

type LogisticRegression struct {
	w     [MLP_N_out][MLP_N_in]float64
	b     [MLP_N_out]float64
	n_in  int
	n_out int
}

func NewLogisticRegression() *LogisticRegression {
	return &LogisticRegression{
		n_in:  MLP_N_in,
		n_out: MLP_N_out,
	}
}

func (l *LogisticRegression) train(x [MLP_MinibatchSize][MLP_N_in]float64, t [MLP_MinibatchSize][MLP_N_out]int, learningRate float64) [MLP_MinibatchSize][MLP_N_out]float64 {
	grad_w := [MLP_N_out][MLP_N_in]float64{}
	grad_b := [MLP_N_out]float64{}
	dy := [MLP_MinibatchSize][MLP_N_out]float64{}

	for n := 0; n < MLP_MinibatchSize; n++ {
		predicted_y := l.output(x[n])
		for j := 0; j < MLP_N_out; j++ {
			dy[n][j] = predicted_y[j] - float64(t[n][j])
			for i := 0; i < MLP_N_in; i++ {
				grad_w[j][i] += dy[n][j] * x[n][i]
			}
			grad_b[j] += dy[n][j]
		}
	}

	for j := 0; j < MLP_N_out; j++ {
		for i := 0; i < MLP_N_in; i++ {
			l.w[j][i] -= learningRate * grad_w[j][i] / float64(MLP_MinibatchSize)
		}
		l.b[j] -= learningRate * grad_b[j] / float64(MLP_MinibatchSize)
	}
	return dy
}

func (l *LogisticRegression) output(x [MLP_N_in]float64) []float64 {
	pre_activation := make([]float64, MLP_N_out)
	for j := 0; j < MLP_N_out; j++ {
		for i := 0; i < MLP_N_in; i++ {
			pre_activation[j] += l.w[j][i] * x[i]
		}
		pre_activation[j] += l.b[j]
	}
	return util.Softmax(pre_activation, MLP_N_out)
}

func (l *LogisticRegression) predict(x [MLP_N_in]float64) [MLP_N_out]int {
	y := l.output(x)
	t := [MLP_N_out]int{}
	arg_max := -1
	var max float64 = 0

	for i := 0; i < MLP_N_out; i++ {
		if max < y[i] {
			max = y[i]
			arg_max = i
		}
	}

	for i := 0; i < MLP_N_out; i++ {
		if i == arg_max {
			t[i] = 1
		} else {
			t[i] = 0
		}
	}
	return t
}
