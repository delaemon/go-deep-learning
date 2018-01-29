package sln

import (
	"fmt"
	"math/rand"

	"github.com/delaemon/go-deep-learning/util"
)

const (
	LRX_N_in          = 2
	LRX_N_out         = 2
	LRX_Patterns      = LRX_N_out
	LRX_Train_n       = 4
	LRX_Test_n        = 4
	LRX_Epochs        = 2000
	LRX_LearningRate  = 0.2
	LRX_MinibatchSize = 1
	LRX_Minibatch_n   = LRX_Train_n / LRX_MinibatchSize
)

type LogisticRegressionXOR struct {
	w [LRX_N_out][LRX_N_in]float64
	b [LRX_N_out]float64
}

func NewLogisticRegressionXOR() *LogisticRegressionXOR {
	return &LogisticRegressionXOR{}
}

func (l *LogisticRegressionXOR) train(x [LRX_MinibatchSize][LRX_N_in]float64, t [LRX_MinibatchSize][LRX_N_out]int, learningRate float64) [LRX_MinibatchSize][LRX_N_out]float64 {
	grad_w := [LRX_N_out][LRX_N_in]float64{}
	grad_b := [LRX_N_out]float64{}
	dy := [LRX_MinibatchSize][LRX_N_out]float64{}

	for n := 0; n < LRX_MinibatchSize; n++ {
		var predicted_y []float64 = l.output(x[n])
		for j := 0; j < LRX_N_out; j++ {
			dy[n][j] = predicted_y[j] - float64(t[n][j])
			for i := 0; i < LRX_N_in; i++ {
				grad_w[j][i] += dy[n][j] * x[n][i]
			}
			grad_b[j] += dy[n][j]
		}
	}

	for j := 0; j < LRX_N_out; j++ {
		for i := 0; i < LRX_N_in; i++ {
			l.w[j][i] -= learningRate * grad_w[j][i] / float64(LRX_MinibatchSize)
		}
		l.b[j] -= learningRate * grad_b[j] / float64(LRX_MinibatchSize)
	}
	return dy
}

func (l *LogisticRegressionXOR) output(x [LRX_N_in]float64) []float64 {
	pre_activation := make([]float64, LRX_N_out)
	for j := 0; j < LRX_N_out; j++ {
		for i := 0; i < LRX_N_in; i++ {
			pre_activation[j] += l.w[j][i] * x[i]
		}
		pre_activation[j] += l.b[j]
	}
	return util.Softmax(pre_activation, LRX_N_out)
}

func (l *LogisticRegressionXOR) predict(x [LRX_N_in]float64) [LRX_N_out]int {
	y := l.output(x)
	t := [LRX_N_out]int{}
	arg_max := -1
	var max float64 = 0

	for i := 0; i < LRX_N_out; i++ {
		if max < y[i] {
			max = y[i]
			arg_max = i
		}
	}

	for i := 0; i < LRX_N_out; i++ {
		if i == arg_max {
			t[i] = 1
		} else {
			t[i] = 0
		}
	}
	return t
}

func (l *LogisticRegressionXOR) Exec() {
	learningRate := LRX_LearningRate
	predicted_t := [LRX_Test_n][LRX_N_out]int{}
	train_x_minibatch := [LRX_Minibatch_n][LRX_MinibatchSize][LRX_N_in]float64{}
	train_t_minibatch := [LRX_Minibatch_n][LRX_MinibatchSize][LRX_N_out]int{}
	minibatch_index := rand.Perm(LRX_Train_n)

	train_x := [LRX_Train_n][LRX_N_in]float64{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}}
	train_t := [LRX_Train_n][LRX_N_out]int{{0, 1}, {1, 0}, {1, 0}, {0, 1}}

	test_x := [LRX_Test_n][LRX_N_in]float64{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}}
	test_t := [LRX_Test_n][LRX_N_out]int{{0, 1}, {1, 0}, {1, 0}, {0, 1}}
	for i := 0; i < LRX_Minibatch_n; i++ {
		for j := 0; j < LRX_MinibatchSize; j++ {
			train_x_minibatch[i][j] = train_x[minibatch_index[i*LRX_MinibatchSize+j]]
			train_t_minibatch[i][j] = train_t[minibatch_index[i*LRX_MinibatchSize+j]]
		}
	}

	classifier := NewLogisticRegressionXOR()
	for epoch := 0; epoch < LRX_Epochs; epoch++ {
		for batch := 0; batch < LRX_Minibatch_n; batch++ {
			classifier.train(train_x_minibatch[batch], train_t_minibatch[batch], learningRate)
		}
		learningRate *= 0.95
	}

	for i := 0; i < LRX_Test_n; i++ {
		predicted_t[i] = classifier.predict(test_x[i])
	}

	// output
	for i := 0; i < LRX_Test_n; i++ {
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
