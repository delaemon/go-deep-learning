package sln

import (
	"fmt"
	"math/rand"

	"github.com/delaemon/go-deep-learning/util"
)

const (
	LR_N_in          = 2
	LR_N_out         = 3
	LR_Patterns      = LR_N_out
	LR_Train_n       = 400 * LR_Patterns
	LR_Test_n        = 60 * LR_Patterns
	LR_Epochs        = 2000
	LR_LearningRate  = 0.2
	LR_MinibatchSize = 50
	LR_Minibatch_n   = LR_Train_n / LR_MinibatchSize
)

type LogisticRegression struct {
	w     [LR_N_out][LR_N_in]float64
	b     [LR_N_out]float64
	n_in  int
	n_out int
}

func NewLogisticRegression() *LogisticRegression {
	return &LogisticRegression{
		n_in:  LR_N_in,
		n_out: LR_N_out,
	}
}

func (l *LogisticRegression) train(x [LR_MinibatchSize][LR_N_in]float64, t [LR_MinibatchSize][LR_N_out]int, learningRate float64) [LR_MinibatchSize][LR_N_out]float64 {
	grad_w := [LR_N_out][LR_N_in]float64{}
	grad_b := [LR_N_out]float64{}
	dy := [LR_MinibatchSize][LR_N_out]float64{}

	for n := 0; n < LR_MinibatchSize; n++ {
		predicted_y := l.output(x[n])
		for j := 0; j < LR_N_out; j++ {
			dy[n][j] = predicted_y[j] - float64(t[n][j])
			for i := 0; i < LR_N_in; i++ {
				grad_w[j][i] += dy[n][j] * x[n][i]
			}
			grad_b[j] += dy[n][j]
		}
	}

	for j := 0; j < LR_N_out; j++ {
		for i := 0; i < LR_N_in; i++ {
			l.w[j][i] -= learningRate * grad_w[j][i] / float64(LR_MinibatchSize)
		}
		l.b[j] -= learningRate * grad_b[j] / float64(LR_MinibatchSize)
	}
	return dy
}

func (l *LogisticRegression) output(x [LR_N_in]float64) []float64 {
	pre_activation := make([]float64, LR_N_out)
	for j := 0; j < LR_N_out; j++ {
		for i := 0; i < LR_N_in; i++ {
			pre_activation[j] += l.w[j][i] * x[i]
		}
		pre_activation[j] += l.b[j]
	}
	return util.Softmax(pre_activation, LR_N_out)
}

func (l *LogisticRegression) predict(x [LR_N_in]float64) [LR_N_out]int {
	y := l.output(x)
	t := [LR_N_out]int{}
	arg_max := -1
	var max float64 = 0

	for i := 0; i < LR_N_out; i++ {
		if max < y[i] {
			max = y[i]
			arg_max = i
		}
	}

	for i := 0; i < LR_N_out; i++ {
		if i == arg_max {
			t[i] = 1
		} else {
			t[i] = 0
		}
	}
	return t
}

func (l *LogisticRegression) search(array [LR_N_out]int, target int) int {
	for k, v := range array {
		if v == target {
			return k
		}
	}
	return -1
}

func (l *LogisticRegression) Exec() {
	learningRate := LR_LearningRate

	train_x := [LR_Train_n][LR_N_in]float64{}
	train_t := [LR_Train_n][LR_N_out]int{}

	test_x := [LR_Test_n][LR_N_in]float64{}
	test_t := [LR_Test_n][LR_N_out]int{}
	predicted_t := [LR_Test_n][LR_N_out]int{}

	train_x_minibatch := [LR_Minibatch_n][LR_MinibatchSize][LR_N_in]float64{}
	train_t_minibatch := [LR_Minibatch_n][LR_MinibatchSize][LR_N_out]int{}
	minibatch_index := rand.Perm(LR_Train_n)

	g1 := util.NewGaussianDistribution(-2.0, 1.0)
	g2 := util.NewGaussianDistribution(2.0, 1.0)
	g3 := util.NewGaussianDistribution(0.0, 1.0)

	for i := 0; i < LR_Train_n/LR_Patterns-1; i++ {
		train_x[i][0] = g1.Random()
		train_x[i][1] = g2.Random()
		train_t[i] = [LR_N_out]int{1, 0, 0}
	}

	for i := 0; i < LR_Test_n/LR_Patterns-1; i++ {
		test_x[i][0] = g1.Random()
		test_x[i][1] = g2.Random()
		test_t[i] = [LR_N_out]int{1, 0, 0}
	}

	for i := LR_Train_n/LR_Patterns - 1; i < LR_Train_n/LR_Patterns*2-1; i++ {
		train_x[i][0] = g2.Random()
		train_x[i][1] = g1.Random()
		train_t[i] = [3]int{0, 1, 0}
	}

	for i := LR_Test_n/LR_Patterns - 1; i < LR_Test_n/LR_Patterns*2-1; i++ {
		test_x[i][0] = g2.Random()
		test_x[i][1] = g1.Random()
		test_t[i] = [3]int{0, 1, 0}
	}

	for i := LR_Train_n/LR_Patterns*2 - 1; i < LR_Train_n; i++ {
		train_x[i][0] = g3.Random()
		train_x[i][1] = g3.Random()
		train_t[i] = [3]int{0, 0, 1}
	}

	for i := LR_Test_n/LR_Patterns*2 - 1; i < LR_Test_n; i++ {
		test_x[i][0] = g3.Random()
		test_x[i][1] = g3.Random()
		test_t[i] = [3]int{0, 0, 1}
	}

	for i := 0; i < LR_Minibatch_n; i++ {
		for j := 0; j < LR_MinibatchSize; j++ {
			train_x_minibatch[i][j] = train_x[minibatch_index[i*LR_MinibatchSize+j]]
			train_t_minibatch[i][j] = train_t[minibatch_index[i*LR_MinibatchSize+j]]
		}
	}

	classifier := NewLogisticRegression()
	for epoch := 0; epoch < LR_Epochs; epoch++ {
		for batch := 0; batch < LR_Minibatch_n; batch++ {
			classifier.train(train_x_minibatch[batch], train_t_minibatch[batch], learningRate)
		}
		learningRate *= 0.95
	}

	for i := 0; i < LR_Test_n; i++ {
		predicted_t[i] = classifier.predict(test_x[i])
	}

	confusion_matrix := [LR_Patterns][LR_Patterns]int{}
	accuracy := 0.0
	precision := [LR_Patterns]float64{}
	recall := [LR_Patterns]float64{}

	for i := 0; i < LR_Test_n; i++ {
		predicted := l.search(predicted_t[i], 1)
		actual := l.search(test_t[i], 1)
		confusion_matrix[actual][predicted] += 1
	}

	for i := 0; i < LR_Patterns; i++ {
		var col float64 = 0
		var row float64 = 0
		for j := 0; j < LR_Patterns; j++ {
			if i == j {
				accuracy += float64(confusion_matrix[i][j])
				precision[i] += float64(confusion_matrix[j][i])
				recall[i] += float64(confusion_matrix[i][j])
			}
			col += float64(confusion_matrix[j][i])
			row += float64(confusion_matrix[i][j])
		}
		precision[i] /= col
		recall[i] /= row
	}
	accuracy /= float64(LR_Test_n)
	fmt.Println("------------------------------")
	fmt.Println("Logistic Regression Evaluation")
	fmt.Println("------------------------------")
	fmt.Printf("Accuracy: %.1f %%\n", accuracy*100)
	fmt.Println("Precision:")
	for i := 0; i < LR_Patterns; i++ {
		fmt.Printf(" class %d: %.1f %%\n", i+1, precision[i]*100)
	}
	fmt.Println("Recall:")
	for i := 0; i < LR_Patterns; i++ {
		fmt.Printf(" class %d: %.1f %%\n", i+1, recall[i]*100)
	}
}
