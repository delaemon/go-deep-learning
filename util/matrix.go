package util

func MatrixInt(row, col int) [][]int {
	matrix := make([][]int, row)
	for i := 0; i < row; i++ {
		matrix[i] = make([]int, col)
	}
	return matrix
}

func MatrixFloat64(row, col int) [][]float64 {
	matrix := make([][]float64, row)
	for i := 0; i < row; i++ {
		matrix[i] = make([]float64, col)
	}
	return matrix
}
