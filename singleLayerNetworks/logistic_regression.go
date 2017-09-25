package network

const n_in int = 2
const n_out int = 3
const patterns int = n_out
const train_n int = 400
const test_n int = 60

type LogisticRegression struct {
}

func NewLogisticRegression() *LogisticRegression {
	return &LogisticRegression{}
}
