package network

import (
	"fmt"

	util "github.com/delaemon/go-deep-learning/util"
)

const n_in int = 2
const train_n int = 100
const test_n int = 200
const epochs int = 2000
const learning_rate float64 = 1

type Perceptrons struct {
	n_in int
	w    []float64
}

func NewPerceptrons() *Perceptrons {
	w := make([]float64, n_in)
	return &Perceptrons{n_in, w}
}

func (p *Perceptrons) train(x [n_in]float64, t int, leaningRate float64) int {
	var classified int = 0
	var c float64 = 0
	for i := 0; i < p.n_in; i++ {
		c += p.w[i] * x[i] * float64(t)
	}
	if c > 0 {
		classified = 1
	} else {
		for i := 0; i < p.n_in; i++ {
			p.w[i] += leaningRate * x[i] * float64(t)
		}
	}
	return classified
}

func (p *Perceptrons) predict(x [n_in]float64) int {
	var preActivation float64 = 0
	for i := 0; i < p.n_in; i++ {
		preActivation += p.w[i] * x[i]
	}
	return util.Step(preActivation)
}

func (p *Perceptrons) Exec() {
	train_x := [train_n][n_in]float64{}
	train_t := [train_n]int{}

	test_x := [test_n][n_in]float64{}
	test_t := [test_n]int{}
	predicted_t := [test_n]int{}

	g1 := util.NewGaussianDistribution(-2.0, 1.0)
	g2 := util.NewGaussianDistribution(2.0, 1.0)

	for i := 0; i < train_n/2-1; i++ {
		train_x[i][0] = g1.Random()
		train_x[i][1] = g2.Random()
		train_t[i] = 1
	}
	for i := 0; i < test_n/2-1; i++ {
		test_x[i][0] = g1.Random()
		test_x[i][1] = g2.Random()
		test_t[i] = 1
	}

	for i := train_n / 2; i < train_n; i++ {
		train_x[i][0] = g2.Random()
		train_x[i][1] = g1.Random()
		train_t[i] = -1
	}
	for i := test_n / 2; i < test_n; i++ {
		test_x[i][0] = g2.Random()
		test_x[i][1] = g1.Random()
		test_t[i] = -1
	}

	// train
	epoch := 0
	classifier := NewPerceptrons()
	for true {
		classified_ := 0
		for i := 0; i < train_n; i++ {
			classified_ += classifier.train(train_x[i], train_t[i], learning_rate)
		}
		if classified_ == train_n {
			break
		}
		epoch++
		if epoch > epochs {
			break
		}
	}

	// test
	for i := 0; i < test_n; i++ {
		predicted_t[i] = classifier.predict(test_x[i])
	}

	confusion_matrix := [2][2]int{}
	var accuracy float64
	var precision float64
	var recall float64

	for i := 0; i < test_n; i++ {
		if predicted_t[i] > 0 {
			if test_t[i] > 0 {
				accuracy += 1
				precision += 1
				recall += 1
				confusion_matrix[0][0] += 1
			} else {
				confusion_matrix[1][0] += 1
			}
		} else {
			if test_t[i] > 0 {
				confusion_matrix[0][1] += 1
			} else {
				accuracy += 1
				confusion_matrix[1][1] += 1
			}
		}
	}

	accuracy /= float64(test_n)
	precision /= float64(confusion_matrix[0][0] + confusion_matrix[1][0])
	recall /= float64(confusion_matrix[0][0] + confusion_matrix[0][1])

	fmt.Println("----------------------------")
	fmt.Println("Perceptrons model evaluation")
	fmt.Println("----------------------------")
	fmt.Printf("Accuracy: %0.1f%% \n", accuracy*100)
	fmt.Printf("Precision: %0.1f%% \n", precision*100)
	fmt.Printf("Recall: %0.1f%% \n", recall*100)
}
