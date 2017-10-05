package network

import (
	"fmt"
	"math/rand"

	"github.com/delaemon/go-deep-learning/util"
)

const n_in int = 2
const n_out int = 2
const patterns int = n_out
const train_n int = 4
const test_n int = 4
const epochs int = 2000
const minibatch_size int = 1
const minibatch_n int = train_n / minibatch_size

type LogisticRegressionXOR struct {
	w             [n_out][n_in]float64
	b             [n_out]float64
	learning_rate float64
}

func NewLogisticRegressionXOR() *LogisticRegressionXOR {
	return &LogisticRegressionXOR{learning_rate: 0.2}
}

func (l *LogisticRegressionXOR) train(x [minibatch_size][n_in]float64, t [minibatch_size][n_out]int) [minibatch_size][n_out]float64 {
	grad_w := [n_out][n_in]float64{}
	grad_b := [n_out]float64{}
	dy := [minibatch_size][n_out]float64{}

	for n := 0; n < minibatch_size; n++ {
		var predicted_y []float64 = l.output(x[n])
		for j := 0; j < n_out; j++ {
			dy[n][j] = predicted_y[j] - float64(t[n][j])
			for i := 0; i < n_in; i++ {
				grad_w[j][i] += dy[n][j] * x[n][i]
			}
			grad_b[j] += dy[n][j]
		}
	}

	for j := 0; j < n_out; j++ {
		for i := 0; i < n_in; i++ {
			l.w[j][i] -= l.learning_rate * grad_w[j][i] / float64(minibatch_size)
		}
		l.b[j] -= l.learning_rate * grad_b[j] / float64(minibatch_size)
	}
	return dy
}

func (l *LogisticRegressionXOR) output(x [n_in]float64) []float64 {
	pre_activation := make([]float64, n_out)
	for j := 0; j < n_out; j++ {
		for i := 0; i < n_in; i++ {
			pre_activation[j] += l.w[j][i] * x[i]
		}
		pre_activation[j] += l.b[j]
	}
	return util.Softmax(pre_activation, n_out)
}

func (l *LogisticRegressionXOR) predict(x [n_in]float64) [n_out]int {
	y := l.output(x)
	t := [n_out]int{}
	arg_max := -1
	var max float64 = 0

	for i := 0; i < n_out; i++ {
		if max < y[i] {
			max = y[i]
			arg_max = i
		}
	}

	for i := 0; i < n_out; i++ {
		if i == arg_max {
			t[i] = 1
		} else {
			t[i] = 0
		}
	}
	return t
}

func (l *LogisticRegressionXOR) Exec() {
	predicted_t := [test_n][n_out]int{}
	train_x_minibatch := [minibatch_n][minibatch_size][n_in]float64{}
	train_t_minibatch := [minibatch_n][minibatch_size][n_out]int{}
	minibatch_index := rand.Perm(train_n)

	train_x := [train_n][n_in]float64{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}}
	train_t := [train_n][n_out]int{{0, 1}, {1, 0}, {1, 0}, {0, 1}}

	test_x := [test_n][n_in]float64{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}}
	test_t := [test_n][n_out]int{{0, 1}, {1, 0}, {1, 0}, {0, 1}}
	for i := 0; i < minibatch_n; i++ {
		for j := 0; j < minibatch_size; j++ {
			train_x_minibatch[i][j] = train_x[minibatch_index[i*minibatch_size+j]]
			train_t_minibatch[i][j] = train_t[minibatch_index[i*minibatch_size+j]]
		}
	}

	classifier := NewLogisticRegressionXOR()
	for epoch := 0; epoch < epochs; epoch++ {
		for batch := 0; batch < minibatch_n; batch++ {
			classifier.train(train_x_minibatch[batch], train_t_minibatch[batch])
		}
		l.learning_rate *= 0.95
	}

	for i := 0; i < test_n; i++ {
		predicted_t[i] = classifier.predict(test_x[i])
	}

	// output
	for i := 0; i < test_n; i++ {
		fmt.Printf("[%.1f,%.1f] -> Prediction: ", test_x[i][0], test_x[i][1])
		if predicted_t[i][0] > predicted_t[i][1] {
			fmt.Printf("Positive, probability = %d", predicted_t[i][0])
		} else {
			fmt.Printf("Negative, probability = %d", predicted_t[i][1])
		}
		fmt.Printf("; Actual: ")
		if test_t[i][0] == 1 {
			fmt.Println("Positive")
		} else {
			fmt.Println("Negative")
		}
	}
}
