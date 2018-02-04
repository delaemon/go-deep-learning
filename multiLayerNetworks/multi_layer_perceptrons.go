package mln

import (
	"fmt"
	"math/rand"
)

const (
	MLP_Patterns      = 2
	MLP_N_in          = 2
	MLP_N_hidden      = 3
	MLP_N_out         = MLP_Patterns
	MLP_Train_N       = 4
	MLP_Test_N        = 4
	MLP_MinibatchSize = 1
	MLP_Minibatch_N   = MLP_Train_N / MLP_MinibatchSize
	MLP_Epochs        = 5000
	MLP_LearningRate  = 0.1
)

type MultiLayerPerceptrons struct {
	n_in          int
	n_hidden      int
	n_out         int
	hiddenLayer   *HiddenLayer
	logisticLayer *LogisticRegression
}

func NewMultiLayerPerceptrons() *MultiLayerPerceptrons {
	return &MultiLayerPerceptrons{
		n_in:          MLP_N_in,
		n_hidden:      MLP_N_hidden,
		n_out:         MLP_N_out,
		hiddenLayer:   NewHiddenLayer("tanh"),
		logisticLayer: NewLogisticRegression(),
	}
}

func (m *MultiLayerPerceptrons) train(x [MLP_MinibatchSize][MLP_N_in]float64, t [MLP_MinibatchSize][MLP_N_out]int) {
	z := [MLP_MinibatchSize][MLP_N_in]float64{}
	dy := [MLP_MinibatchSize][MLP_N_out]float64{}

	for n := 0; n < MLP_MinibatchSize; n++ {
		z[n] = m.hiddenLayer.forward(x[n])
	}

	dy = m.logisticLayer.train(z, t, MLP_LearningRate)

	m.hiddenLayer.backward(x, z, dy, m.logisticLayer.w)
}

func (m *MultiLayerPerceptrons) predict(x [MLP_N_in]float64) [MLP_N_out]int {
	z := m.hiddenLayer.output(x)
	return m.logisticLayer.predict(z)
}

func (m *MultiLayerPerceptrons) Exec() {
	predicted_t := [MLP_Test_N][MLP_N_out]int{}
	train_x_minibatch := [MLP_Minibatch_N][MLP_MinibatchSize][MLP_N_in]float64{}
	train_t_minibatch := [MLP_Minibatch_N][MLP_MinibatchSize][MLP_N_out]int{}
	minibatch_index := rand.Perm(MLP_Train_N)

	train_x := [MLP_Train_N][MLP_N_in]float64{
		{0.0, 0.0},
		{0.0, 1.0},
		{1.0, 0.0},
		{1.0, 1.0},
	}

	train_t := [MLP_Train_N][MLP_N_out]int{
		{0, 1},
		{1, 0},
		{1, 0},
		{0, 1},
	}

	test_x := [MLP_Test_N][MLP_N_in]float64{
		{0.0, 0.0},
		{0.0, 1.0},
		{1.0, 0.0},
		{1.0, 1.0},
	}

	test_t := [MLP_Test_N][MLP_N_out]int{
		{0, 1},
		{1, 0},
		{1, 0},
		{0, 1},
	}

	for i := 0; i < MLP_Minibatch_N; i++ {
		for j := 0; j < MLP_MinibatchSize; j++ {
			train_x_minibatch[i][j] = train_x[minibatch_index[i*MLP_MinibatchSize+j]]
			train_t_minibatch[i][j] = train_t[minibatch_index[i*MLP_MinibatchSize+j]]
		}
	}

	classifier := NewMultiLayerPerceptrons()

	for epoch := 0; epoch < MLP_Epochs; epoch++ {
		for batch := 0; batch < MLP_Minibatch_N; batch++ {
			classifier.train(train_x_minibatch[batch], train_t_minibatch[batch])
		}
	}

	for i := 0; i < MLP_Test_N; i++ {
		predicted_t[i] = classifier.predict(test_x[i])
	}

	confusion_matrix := [MLP_Patterns][MLP_Patterns]int{}
	accuracy := 0.0
	precision := [MLP_Patterns]float64{}
	recall := [MLP_Patterns]float64{}

	for i := 0; i < MLP_Test_N; i++ {
		predicted_ := predicted_t[i][1]
		actual_ := test_t[i][1]
		confusion_matrix[actual_][predicted_] += 1
	}

	for i := 0; i < MLP_Patterns; i++ {
		col_ := 0.0
		row_ := 0.0
		for j := 0; j < MLP_Patterns; j++ {
			if i == j {
				accuracy += float64(confusion_matrix[i][j])
				precision[i] += float64(confusion_matrix[j][i])
				recall[i] += float64(confusion_matrix[i][j])
			}
			col_ += float64(confusion_matrix[j][i])
			row_ += float64(confusion_matrix[i][j])
		}
		precision[i] /= col_
		recall[i] /= row_
	}
	accuracy /= MLP_Test_N
	fmt.Println("--------------------")
	fmt.Println("MLP model evaluation")
	fmt.Println("--------------------")
	fmt.Printf("Accuracy: %.1f %%\n", accuracy*100)
	fmt.Println("Precision:")
	for i := 0; i < MLP_Patterns; i++ {
		fmt.Printf(" class %d: %.1f %%\n", i+1, precision[i]*100)
	}
	fmt.Println("Recall:")
	for i := 0; i < MLP_Patterns; i++ {
		fmt.Printf(" class %d: %.1f %%\n", i+1, recall[i]*100)
	}
}
