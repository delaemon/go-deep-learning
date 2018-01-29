package main

import sln "github.com/delaemon/go-deep-learning/singleLayerNetworks"

func main() {
	ml := sln.NewPerceptrons()
	//ml := sln.NewLogisticRegression()
	//ml := sln.NewLogisticRegressionXOR()
	ml.Exec()
}
