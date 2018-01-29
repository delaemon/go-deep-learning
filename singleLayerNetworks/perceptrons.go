package sln

import (
	"fmt"

	util "github.com/delaemon/go-deep-learning/util"
)

const (
	P_N_in         = 2
	P_Train_n      = 100
	P_Test_n       = 200
	P_Epochs       = 2000
	P_LearningRate = 1.0
)

type Perceptrons struct {
	w []float64
}

func NewPerceptrons() *Perceptrons {
	return &Perceptrons{
		w: make([]float64, P_N_in),
	}
}

func (p *Perceptrons) train(x []float64, t int, leaningRate float64) int {
	var classified int = 0
	var c float64 = 0
	for i := 0; i < P_N_in; i++ {
		c += p.w[i] * x[i] * float64(t)
	}
	if c > 0 {
		classified = 1
	} else {
		for i := 0; i < P_N_in; i++ {
			p.w[i] += leaningRate * x[i] * float64(t)
		}
	}
	return classified
}

func (p *Perceptrons) predict(x []float64) int {
	var preActivation float64 = 0
	for i := 0; i < P_N_in; i++ {
		preActivation += p.w[i] * x[i]
	}
	return util.Step(preActivation)
}

func (p *Perceptrons) Exec() {
	train_x := make([][]float64, P_Train_n)
	for i := 0; i < P_Train_n; i++ {
		train_x[i] = make([]float64, P_N_in)
	}
	train_t := make([]int, P_Train_n)

	test_x := make([][]float64, P_Test_n)
	for i := 0; i < P_Test_n; i++ {
		test_x[i] = make([]float64, P_N_in)
	}
	test_t := make([]int, P_Test_n)
	predicted_t := make([]int, P_Test_n)

	g1 := util.NewGaussianDistribution(-2.0, 1.0)
	g2 := util.NewGaussianDistribution(2.0, 1.0)

	for i := 0; i < P_Train_n/2-1; i++ {
		train_x[i][0] = g1.Random()
		train_x[i][1] = g2.Random()
		train_t[i] = 1
	}
	for i := 0; i < P_Test_n/2-1; i++ {
		test_x[i][0] = g1.Random()
		test_x[i][1] = g2.Random()
		test_t[i] = 1
	}

	for i := P_Train_n / 2; i < P_Train_n; i++ {
		train_x[i][0] = g2.Random()
		train_x[i][1] = g1.Random()
		train_t[i] = -1
	}
	for i := P_Test_n / 2; i < P_Test_n; i++ {
		test_x[i][0] = g2.Random()
		test_x[i][1] = g1.Random()
		test_t[i] = -1
	}

	// train
	epoch := 0
	classifier := NewPerceptrons()
	for true {
		classified := 0
		for i := 0; i < P_Train_n; i++ {
			classified += classifier.train(train_x[i], train_t[i], P_LearningRate)
		}
		if classified == P_Train_n {
			break
		}
		epoch++
		if epoch > P_Epochs {
			break
		}
	}

	// test
	for i := 0; i < P_Test_n; i++ {
		predicted_t[i] = classifier.predict(test_x[i])
	}

	confusion_matrix := [2][2]int{}
	var accuracy float64
	var precision float64
	var recall float64

	for i := 0; i < P_Test_n; i++ {
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

	accuracy /= float64(P_Test_n)
	precision /= float64(confusion_matrix[0][0] + confusion_matrix[1][0])
	recall /= float64(confusion_matrix[0][0] + confusion_matrix[0][1])

	fmt.Println("----------------------------")
	fmt.Println("Perceptrons model evaluation")
	fmt.Println("----------------------------")
	fmt.Printf("Accuracy: %0.1f%% \n", accuracy*100)
	fmt.Printf("Precision: %0.1f%% \n", precision*100)
	fmt.Printf("Recall: %0.1f%% \n", recall*100)
}
