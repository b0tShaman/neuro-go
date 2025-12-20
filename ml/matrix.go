package ml

import (
	"bytes"
	"encoding/gob"
	"math"
	"math/rand/v2"

	"gonum.org/v1/gonum/mat"
)

// Matrix represents a dense matrix with a flat data slice for performance.
type Matrix struct {
	rows, cols int
	data       []float64
	dense      *mat.Dense
}

// -------- CONSTRUCTORS ------- //
func NewMatrix(rows, cols int) *Matrix {
	data := make([]float64, rows*cols)
	return &Matrix{
		rows:  rows,
		cols:  cols,
		data:  data,
		dense: mat.NewDense(rows, cols, data),
	}
}

func NewMatrixFromSlice(rows, cols int, data []float64) *Matrix {
	if len(data) != rows*cols {
		panic("Slice length mismatch")
	}

	return &Matrix{
		rows:  rows,
		cols:  cols,
		data:  data,
		dense: mat.NewDense(rows, cols, data),
	}
}

// ------- MATRIX METHODS ------ //
func (m *Matrix) GobEncode() ([]byte, error) {
	w := new(bytes.Buffer)
	encoder := gob.NewEncoder(w)
	if err := encoder.Encode(m.rows); err != nil {
		return nil, err
	}
	if err := encoder.Encode(m.cols); err != nil {
		return nil, err
	}
	if err := encoder.Encode(m.data); err != nil {
		return nil, err
	}
	return w.Bytes(), nil
}

func (m *Matrix) GobDecode(buf []byte) error {
	r := bytes.NewBuffer(buf)
	decoder := gob.NewDecoder(r)
	if err := decoder.Decode(&m.rows); err != nil {
		return err
	}
	if err := decoder.Decode(&m.cols); err != nil {
		return err
	}
	if err := decoder.Decode(&m.data); err != nil {
		return err
	}

	// Re-create the wrapper after loading data
	m.dense = mat.NewDense(m.rows, m.cols, m.data)

	return nil
}

func (m *Matrix) Randomize() {
	scale := math.Sqrt(2.0 / float64(m.rows))
	for i := range m.data {
		m.data[i] = rand.NormFloat64() * scale
	}
}

func (m *Matrix) RandomizeXavier() {
	// limit = sqrt(6 / (fan_in + fan_out))
	limit := math.Sqrt(6.0 / float64(m.rows+m.cols))
	for i := range m.data {
		// Uniform distribution between -limit and limit
		m.data[i] = (rand.Float64()*2 - 1) * limit
	}
}

func (m *Matrix) Reset() {
	for i := range m.data {
		m.data[i] = 0.0
	}
}

func (m *Matrix) Add(b *Matrix) {
	m.dense.Add(m.dense, b.dense)
}

func (m *Matrix) Subtract(b *Matrix) {
	m.dense.Sub(m.dense, b.dense)
}

func (m *Matrix) AddVector(v *Matrix) {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.data[i*m.cols+j] += v.data[j]
		}
	}
}

func (m *Matrix) ApplyRelu() {
	for i, v := range m.data {
		if v < 0 {
			m.data[i] = 0
		}
	}
}

func (m *Matrix) ApplySigmoid() {
	for i, v := range m.data {
		m.data[i] = 1.0 / (1.0 + math.Exp(-v))
	}
}

func (m *Matrix) ApplyFunc(fn func(float64) float64) {
	for i := range m.data {
		m.data[i] = fn(m.data[i])
	}
}

// ------ UTILITY FUNCTIONS ------
func MatMul(a, b mat.Matrix, out *Matrix) {
	out.dense.Mul(a, b)
}

// MatMul using pure go (no BLAS)
func MatMulGo(a, b, out *Matrix) {
	const blockSize = 64
	if a.cols != b.rows || out.rows != a.rows || out.cols != b.cols {
		panic("Shape mismatch")
	}
	out.Reset()
	for i := 0; i < a.rows; i += blockSize {
		for j := 0; j < b.cols; j += blockSize {
			for k := 0; k < a.cols; k += blockSize {
				iMax, jMax, kMax := i+blockSize, j+blockSize, k+blockSize
				if iMax > a.rows {
					iMax = a.rows
				}
				if jMax > b.cols {
					jMax = b.cols
				}
				if kMax > a.cols {
					kMax = a.cols
				}
				for ii := i; ii < iMax; ii++ {
					rowOffsetOut := ii * out.cols
					for kk := k; kk < kMax; kk++ {
						scalar := a.data[ii*a.cols+kk]
						rowOffsetB := kk * b.cols
						for jj := j; jj < jMax; jj++ {
							out.data[rowOffsetOut+jj] += scalar * b.data[rowOffsetB+jj]
						}
					}
				}
			}
		}
	}
}
