package dnn

const (
	Train_n_each      = 200
	Test_n_each       = 2
	N_visible_each    = 4
	P_noise_trainging = 0.05
	P_noise_test      = 0.25
	Patterns          = 3
	Train_n           = Train_n_each * Patterns
	Test_n            = Test_n_each * Patterns
	N_visible         = N_visible_each * Patterns
	N_hidden          = 6
	Epochs            = 1000
	LearningRate      = 0.2
	MinibatchSize     = 10
	Minibatch_n       = Train_n / MinibatchSize
)

type RestrictedBoltzmannMachines struct {
	w     [N_hidden][N_visible]float64
	hbias [N_hidden]float64
	vbias [N_visible]float64
}

func Exec() {
}

func NewRestrictedBoltzmannMachines() *RestrictedBoltzmannMachines {
	return &RestrictedBoltzmannMachines{}
}
