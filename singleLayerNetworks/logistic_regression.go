package network

import (
	"fmt"
	"math/rand"

	"github.com/delaemon/go-deep-learning/util"
)

const n_in int = 2
const n_out int = 3
const patterns int = n_out
const train_n int = 400
const test_n int = 60
const epochs int = 2000
const minibatch_size int = 50
const minibatch_n int = train_n / minibatch_size

type LogisticRegression struct {
	w             [n_out][n_in]float64
	b             [n_out]float64
	learning_rate float64
}

func NewLogisticRegression() *LogisticRegression {
	return &LogisticRegression{learning_rate: 0.2}
}

func (l *LogisticRegression) train(x [minibatch_size][n_in]float64, t [minibatch_size][n_out]int) [minibatch_size][n_out]float64 {
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

func (l *LogisticRegression) output(x [n_in]float64) []float64 {
	pre_activation := make([]float64, n_out)
	for j := 0; j < n_out; j++ {
		for i := 0; i < n_in; i++ {
			pre_activation[j] += l.w[j][i] * x[i]
		}
		pre_activation[j] += l.b[j]
	}
	return util.Softmax(pre_activation, n_out)
}

func (l *LogisticRegression) predict(x [n_in]float64) [n_out]int {
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

func (l *LogisticRegression) search(array [patterns]int, target int) int {
	for k, v := range array {
		if v == target {
			return k
		}
	}
	return -1
}

func (l *LogisticRegression) Exec() {
	train_x := [train_n][n_in]float64{}
	train_t := [train_n][n_out]int{}

	test_x := [test_n][n_in]float64{}
	test_t := [test_n][n_out]int{}
	predicted_t := [test_n][n_out]int{}

	train_x_minibatch := [minibatch_n][minibatch_size][n_in]float64{}
	train_t_minibatch := [minibatch_n][minibatch_size][n_out]int{}
	minibatch_index := rand.Perm(train_n)

	g1 := util.NewGaussianDistribution(-2.0, 1.0)
	g2 := util.NewGaussianDistribution(2.0, 1.0)
	g3 := util.NewGaussianDistribution(0.0, 1.0)

	for i := 0; i < train_n/patterns-1; i++ {
		train_x[i][0] = g1.Random()
		train_x[i][1] = g2.Random()
		train_t[i] = [3]int{1, 0, 0}
	}

	for i := 0; i < test_n/patterns-1; i++ {
		test_x[i][0] = g1.Random()
		test_x[i][1] = g2.Random()
		test_t[i] = [3]int{1, 0, 0}
	}

	for i := train_n/patterns - 1; i < train_n/patterns*2-1; i++ {
		train_x[i][0] = g2.Random()
		train_x[i][1] = g1.Random()
		train_t[i] = [3]int{0, 1, 0}
	}

	for i := test_n/patterns - 1; i < test_n/patterns*2-1; i++ {
		test_x[i][0] = g2.Random()
		test_x[i][1] = g1.Random()
		test_t[i] = [3]int{0, 1, 0}
	}

	for i := train_n/patterns*2 - 1; i < train_n; i++ {
		train_x[i][0] = g3.Random()
		train_x[i][1] = g3.Random()
		train_t[i] = [3]int{0, 0, 1}
	}

	for i := test_n/patterns*2 - 1; i < test_n; i++ {
		test_x[i][0] = g3.Random()
		test_x[i][1] = g3.Random()
		test_t[i] = [3]int{0, 0, 1}
	}

	for i := 0; i < minibatch_n; i++ {
		for j := 0; j < minibatch_size; j++ {
			train_x_minibatch[i][j] = train_x[minibatch_index[i*minibatch_size+j]]
			train_t_minibatch[i][j] = train_t[minibatch_index[i*minibatch_size+j]]
		}
	}

	classifier := NewLogisticRegression()
	for epoch := 0; epoch < epochs; epoch++ {
		for batch := 0; batch < minibatch_n; batch++ {
			classifier.train(train_x_minibatch[batch], train_t_minibatch[batch])
		}
		l.learning_rate *= 0.95
	}

	for i := 0; i < test_n; i++ {
		predicted_t[i] = classifier.predict(test_x[i])
	}

	confusion_matrix := [patterns][patterns]int{}
	accuracy := 0.0
	precision := [patterns]float64{}
	recall := [patterns]float64{}

	for i := 0; i < test_n; i++ {
		predicted := l.search(predicted_t[i], 1)
		actual := l.search(test_t[i], 1)
		confusion_matrix[actual][predicted] += 1
	}

	for i := 0; i < patterns; i++ {
		var col float64 = 0
		var row float64 = 0
		for j := 0; j < patterns; j++ {
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
	accuracy /= float64(test_n)
	fmt.Println("------------------------------")
	fmt.Println("Logistic Regression Evaluation")
	fmt.Println("------------------------------")
	fmt.Printf("Accuracy: %.1f %%\n", accuracy*100)
	fmt.Println("Precision:")
	for i := 0; i < patterns; i++ {
		fmt.Printf(" class %d: %.1f %%\n", i+1, precision[i]*100)
	}
	fmt.Println("Recall:")
	for i := 0; i < patterns; i++ {
		fmt.Printf(" class %d: %.1f %%\n", i+1, recall[i]*100)
	}
}
