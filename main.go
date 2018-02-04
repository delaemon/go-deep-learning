package main

import (
	//	sln "github.com/delaemon/go-deep-learning/singleLayerNetworks"
	mln "github.com/delaemon/go-deep-learning/multiLayerNetworks"
)

func main() {
	//ml := sln.NewPerceptrons()
	//ml := sln.NewLogisticRegression()
	//ml := sln.NewLogisticRegressionXOR()
	ml := mln.NewMultiLayerPerceptrons()
	ml.Exec()
}
