package main

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"image/jpeg"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"runtime"
	"sync"
	"syscall"
	"time"

	"github.com/b0tShaman/go-rout-net/data"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// --- Matrix Library ---

// Matrix represents a dense matrix with a flat data slice for performance.
type Matrix struct {
	rows, cols int
	data       []float64
	dense      *mat.Dense
}

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

func (m *Matrix) Reset() {
	for i := range m.data {
		m.data[i] = 0.0
	}
}

// Add performs element-wise addition: m = m + b
func (m *Matrix) Add(b *Matrix) {
	// Safety check (optional for raw speed, but good for debugging)
	if len(m.data) != len(b.data) {
		panic("Matrix shape mismatch in Add")
	}

	// floats.Add(dst, src) -> dst[i] += src[i]
	floats.Add(m.data, b.data)
}

// Subtract performs element-wise subtraction: m = m - b
func (m *Matrix) Subtract(b *Matrix) {
	if len(m.data) != len(b.data) {
		panic("Matrix shape mismatch in Subtract")
	}

	// floats.Sub(dst, src) -> dst[i] -= src[i]
	floats.Sub(m.data, b.data)
}

// Transpose transposes 'm' into 'dst'.
// 'dst' must be pre-allocated with dimensions (m.cols x m.rows).
func (m *Matrix) Transpose(dst *Matrix) {
	// Safety check (can be removed in production for speed)
	if dst.rows != m.cols || dst.cols != m.rows {
		panic(fmt.Sprintf("Shape mismatch in Transpose: Src(%d,%d) vs Dst(%d,%d)",
			m.rows, m.cols, dst.rows, dst.cols))
	}

	// Standard cache-unfriendly loop (hard to optimize without blocking)
	for i := 0; i < m.rows; i++ {
		rowOffset := i * m.cols
		for j := 0; j < m.cols; j++ {
			dst.data[j*dst.cols+i] = m.data[rowOffset+j]
		}
	}
}

func (m *Matrix) AddVector(v *Matrix) {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.data[i*m.cols+j] += v.data[j]
		}
	}
}

func (m *Matrix) ApplyFunc(fn func(float64) float64) {
	for i := range m.data {
		m.data[i] = fn(m.data[i])
	}
}

// Optimized MatMul
func MatMul(a, b, out *Matrix) {
	// No allocations here anymore
	out.dense.Mul(a.dense, b.dense)
}

// MatMul using pure go (no BLAS)
const blockSize = 64

func MatMulGo(a, b, out *Matrix) {
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

func flatten(input [][]float64) []float64 {
	if len(input) == 0 {
		return nil
	}
	rows, cols := len(input), len(input[0])
	flat := make([]float64, rows*cols)
	for i, row := range input {
		copy(flat[i*cols:], row)
	}
	return flat
}

func Relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func ReluDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// SoftmaxRow applies softmax to each row of the matrix.
func SoftmaxRow(m *Matrix) {
	for i := 0; i < m.rows; i++ {
		maxVal := -math.MaxFloat64
		for j := 0; j < m.cols; j++ {
			if m.data[i*m.cols+j] > maxVal {
				maxVal = m.data[i*m.cols+j]
			}
		}
		sum := 0.0
		for j := 0; j < m.cols; j++ {
			val := math.Exp(m.data[i*m.cols+j] - maxVal)
			m.data[i*m.cols+j] = val
			sum += val
		}
		for j := 0; j < m.cols; j++ {
			m.data[i*m.cols+j] /= sum
		}
	}
}

// --- NEW CONFIGURATION INTERFACE ---

type ActivationType int

const (
	ActNone ActivationType = iota
	ActRelu
	ActSoftmax
)

// LayerConfig holds the blueprint for a layer
type LayerConfig struct {
	Neurons    int
	IsInput    bool
	Activation ActivationType
}

type Layer struct {
	// Parameters (Shared among workers)
	Weights *Matrix
	Biases  *Matrix

	// Forward State (Unique to worker)
	Z *Matrix
	A *Matrix

	// Backward Scratchpads (Unique to worker)
	// We pre-allocate these to avoid allocating them 60,000 times!
	dZ       *Matrix // Error signal for this layer
	AT       *Matrix // Transposed A (for dW calc)
	WeightsT *Matrix // Transposed Weights (for previous dZ calc)

	// Configuration
	ActType ActivationType // Store the activation type here
}

// Input defines the entry point dimensions
func Input(size int) LayerConfig {
	return LayerConfig{
		Neurons:    size,
		IsInput:    true,
		Activation: ActNone,
	}
}

// Dense defines a fully connected layer.
// By default, we use ReLU. The builder logic will override the LAST layer to Softmax
// if not specified, or we can handle it explicitly.
func Dense(size int) LayerConfig {
	return LayerConfig{
		Neurons:    size,
		IsInput:    false,
		Activation: ActRelu, // Default for hidden layers
	}
}

// GradientSet holds the calculated gradients for one layer
type GradientSet struct {
	dW *Matrix
	db *Matrix
}

// --- Neural Network ---

type NeuralNetwork struct {
	Layers       []*Layer
	LearningRate float64
	InputT       *Matrix
	InputBuf     *Matrix // <--- Reusable Input Wrapper
}

// NewNetwork is the new Constructor using the variadic interface
func NewNetwork(lr float64, configs ...LayerConfig) *NeuralNetwork {
	if len(configs) < 2 {
		panic("Network must have at least Input and one Output layer")
	}
	if !configs[0].IsInput {
		panic("First layer must be Input()")
	}

	nn := &NeuralNetwork{LearningRate: lr}

	// Track the output size of the previous layer
	prevOutputSize := configs[0].Neurons

	for i := 1; i < len(configs); i++ {
		cfg := configs[i]

		// If it's the very last layer, force Softmax (or let user decide,
		// but for this specific code which uses CrossEntropy logic, Softmax is required).
		act := cfg.Activation
		if i == len(configs)-1 {
			act = ActSoftmax
		}

		layer := &Layer{
			Weights: NewMatrix(prevOutputSize, cfg.Neurons),
			Biases:  NewMatrix(1, cfg.Neurons),
			ActType: act,
		}

		layer.Weights.Randomize()
		nn.Layers = append(nn.Layers, layer)

		// Update for next iteration
		prevOutputSize = cfg.Neurons
	}

	return nn
}

func (nn *NeuralNetwork) CloneStructure() *NeuralNetwork {
	newNN := &NeuralNetwork{
		LearningRate: nn.LearningRate,
		Layers:       make([]*Layer, len(nn.Layers)),
	}
	for i, l := range nn.Layers {
		newNN.Layers[i] = &Layer{
			Weights: l.Weights,
			Biases:  l.Biases,
			ActType: l.ActType, // Copy activation type
		}
	}
	return newNN
}

func (nn *NeuralNetwork) InitializeBuffers(batchSize int) {
	inputDim := nn.Layers[0].Weights.rows

	// FIX: Manually initialize 'dense' here
	data := make([]float64, batchSize*inputDim)
	nn.InputBuf = &Matrix{
		rows:  batchSize,
		cols:  inputDim,
		data:  data,
		dense: mat.NewDense(batchSize, inputDim, data), // <--- ADD THIS
	}

	nn.InputT = NewMatrix(inputDim, batchSize)

	for _, layer := range nn.Layers {
		// These use NewMatrix, so they are safe IF NewMatrix is updated
		layer.Z = NewMatrix(batchSize, layer.Weights.cols)
		layer.A = NewMatrix(batchSize, layer.Weights.cols)
		layer.dZ = NewMatrix(batchSize, layer.Weights.cols)
		layer.AT = NewMatrix(layer.Weights.cols, batchSize)
		layer.WeightsT = NewMatrix(layer.Weights.cols, layer.Weights.rows)
	}
}

// Optimized Forward with Activation Switch
func (nn *NeuralNetwork) Forward(input *Matrix) {
	activation := input
	for _, layer := range nn.Layers {
		MatMul(activation, layer.Weights, layer.Z)
		layer.Z.AddVector(layer.Biases)

		// Direct copy Z -> A
		copy(layer.A.data, layer.Z.data)

		// Apply Activation based on ActType
		switch layer.ActType {
		case ActSoftmax:
			SoftmaxRow(layer.A)
		case ActRelu:
			layer.A.ApplyFunc(Relu)
		case ActNone:
			// Linear activation (pass through)
		}

		activation = layer.A
	}
}

// ComputeGradients remains largely the same, assuming CrossEntropy+Softmax at the end.
func (nn *NeuralNetwork) ComputeGradients(input *Matrix, Y []float64, grads []GradientSet) (float64, float64) {
	loss, acc := nn.ComputeLossAndAccuracy(Y)

	batchSize := float64(input.rows)
	scale := 1.0 / batchSize

	lastLayerIdx := len(nn.Layers) - 1
	lastLayer := nn.Layers[lastLayerIdx]

	// 1. Output Error (Assumes Softmax + Cross Entropy)
	copy(lastLayer.dZ.data, lastLayer.A.data)
	for i, classLabel := range Y {
		idx := i*lastLayer.dZ.cols + int(classLabel)
		lastLayer.dZ.data[idx] -= 1.0
	}

	// 2. Backprop
	for i := lastLayerIdx; i >= 0; i-- {
		layer := nn.Layers[i]
		var A_prev *Matrix
		if i == 0 {
			A_prev = input
		} else {
			A_prev = nn.Layers[i-1].A
		}

		// dW
		if i > 0 {
			prevLayer := nn.Layers[i-1]
			A_prev.Transpose(prevLayer.AT)
			MatMul(prevLayer.AT, layer.dZ, grads[i].dW)
		} else {
			A_prev.Transpose(nn.InputT)
			MatMul(nn.InputT, layer.dZ, grads[i].dW)
		}

		// db
		grads[i].db.Reset()
		dZData := layer.dZ.data
		dbData := grads[i].db.data
		cols := layer.dZ.cols
		for r := 0; r < layer.dZ.rows; r++ {
			rowOffset := r * cols
			for c := 0; c < cols; c++ {
				dbData[c] += dZData[rowOffset+c]
			}
		}

		// Scale
		grads[i].dW.ApplyFunc(func(v float64) float64 { return v * scale })
		grads[i].db.ApplyFunc(func(v float64) float64 { return v * scale })

		// dZ Prev
		if i > 0 {
			prevLayer := nn.Layers[i-1]
			layer.Weights.Transpose(layer.WeightsT)
			MatMul(layer.dZ, layer.WeightsT, prevLayer.dZ)

			// Derivative Application depends on Previous Layer's Activation
			// Since we defaulted internal layers to Relu, we use Relu Derivative.
			// If you add other activations, add a switch here too.
			zData := prevLayer.Z.data
			dZPrevData := prevLayer.dZ.data
			for k := range dZPrevData {
				if zData[k] <= 0 {
					dZPrevData[k] = 0
				}
			}
		}
	}
	return loss, acc
}

// SaveToFile saves the neural network weights and biases to a file.
// Save/Load logic needs to store the ActivationType now
func (nn *NeuralNetwork) SaveToFile(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	encoder := gob.NewEncoder(file)

	type LayerData struct {
		Weights *Matrix
		Biases  *Matrix
		ActType ActivationType
	}
	type NetworkData struct {
		LayerDatas   []LayerData
		LearningRate float64
	}

	ld := make([]LayerData, len(nn.Layers))
	for i, l := range nn.Layers {
		ld[i] = LayerData{Weights: l.Weights, Biases: l.Biases, ActType: l.ActType}
	}

	return encoder.Encode(NetworkData{LayerDatas: ld, LearningRate: nn.LearningRate})
}

func (nn *NeuralNetwork) LoadFromFile(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	decoder := gob.NewDecoder(file)

	type LayerData struct {
		Weights *Matrix
		Biases  *Matrix
		ActType ActivationType
	}
	type NetworkData struct {
		LayerDatas   []LayerData
		LearningRate float64
	}

	var data NetworkData
	if err := decoder.Decode(&data); err != nil {
		return err
	}

	nn.Layers = make([]*Layer, len(data.LayerDatas))
	for i, ld := range data.LayerDatas {
		nn.Layers[i] = &Layer{
			Weights: ld.Weights,
			Biases:  ld.Biases,
			ActType: ld.ActType,
		}
	}
	nn.LearningRate = data.LearningRate
	return nil
}

func (nn *NeuralNetwork) ComputeLossAndAccuracy(Y []float64) (float64, float64) {
	output := nn.Layers[len(nn.Layers)-1].A
	totalLoss := 0.0
	correctCount := 0
	epsilon := 1e-15

	for i := 0; i < output.rows; i++ {
		maxProb := -1.0
		predictedClass := -1
		targetClass := Y[i]

		for j := 0; j < output.cols; j++ {
			prob := output.data[i*output.cols+j]
			if prob > maxProb {
				maxProb = prob
				predictedClass = j
			}
			if j == int(targetClass) {
				totalLoss += -math.Log(prob + epsilon)
			}
		}
		if predictedClass == int(targetClass) {
			correctCount++
		}
	}
	return totalLoss / float64(output.rows), float64(correctCount) / float64(output.rows)
}

func (nn *NeuralNetwork) UpdateWeights(aggregatedGrads []GradientSet, splitScale float64) {
	effectiveLR := nn.LearningRate * splitScale

	for i, layer := range nn.Layers {
		gradW, gradB := aggregatedGrads[i].dW, aggregatedGrads[i].db

		// 1. Update Weights
		// Logic: Weights = Weights + (-effectiveLR * gradW)
		// This runs in one SIMD pass over the whole slice.
		floats.AddScaledTo(layer.Weights.data, layer.Weights.data, -effectiveLR, gradW.data)

		// 2. Update Biases
		// Logic: Biases = Biases + (-effectiveLR * gradB)
		floats.AddScaledTo(layer.Biases.data, layer.Biases.data, -effectiveLR, gradB.data)
	}
}

// Predict takes a flattened image (1D slice), passes it through the network,
// and returns the predicted Class (0-9) and the Confidence (probability).
func (nn *NeuralNetwork) Predict(inputData []float64) (int, float64) {
	// 1. Validation
	inputSize := nn.Layers[0].Weights.rows
	if len(inputData) != inputSize {
		panic(fmt.Sprintf("Input size mismatch. Expected %d, got %d", inputSize, len(inputData)))
	}

	// Check if buffers exist and match the batch size (1)
	if nn.Layers[0].Z == nil || nn.Layers[0].Z.rows != 1 {
		nn.InitializeBuffers(1)
	}

	// 2. Create Matrix View (1 Row, N Cols)
	// We treat the single image as a Batch of size 1.
	inputMat := NewMatrixFromSlice(1, inputSize, inputData)

	// 3. Run Forward Pass
	// This updates the Master NN's internal 'A' matrices.
	nn.Forward(inputMat)

	// 4. Read Output
	// The result is in the last layer's Activation matrix (A).
	lastLayer := nn.Layers[len(nn.Layers)-1]
	probabilities := lastLayer.A.data // This is a slice of size 10 (for MNIST)

	// 5. Argmax (Find highest probability)
	bestClass := -1
	maxProb := -1.0

	for i, prob := range probabilities {
		if prob > maxProb {
			maxProb = prob
			bestClass = i
		}
	}

	return bestClass, maxProb
}

func convertJpg1D(path string) ([]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	img, err := jpeg.Decode(f)
	if err != nil {
		return nil, err
	}

	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	out := make([]float64, 0, w*h)

	for y := range h {
		for x := range w {
			r, g, b, _ := img.At(x, y).RGBA()

			gray := 0.299*float64(r>>8) +
				0.587*float64(g>>8) +
				0.114*float64(b>>8)

			out = append(out, gray)
		}
	}

	return out, nil
}

// NewIndexList creates a slice [0, 1, 2, ..., size-1]
func NewIndexList(size int) []int {
	indices := make([]int, size)
	for i := range indices {
		indices[i] = i
	}
	return indices
}

// ShuffleIndices performs Fisher-Yates on the index slice only.
// This is extremely fast compared to shuffling the float data.
func ShuffleIndices(indices []int) {
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})
}

// Gather copies specific rows from the global storage into the worker's local buffer.
// This allows the worker to have a contiguous matrix for efficient MatMul,
// without needing to reshuffle the global array.
func Gather(
	batchIndices []int, // The random indices for this worker (e.g., 16 items)
	globalX []float64, // The massive immutable data
	globalY []float64, // The massive immutable labels
	inputDim int, // e.g., 784
	destX *Matrix, // The worker's local input matrix
	destY []float64, // The worker's local label slice
) {
	rowSize := inputDim

	for localRowIdx, realDataIdx := range batchIndices {
		// 1. Copy Label
		destY[localRowIdx] = globalY[realDataIdx]

		// 2. Copy Input Vector
		// We calculate where this specific image lives in the massive global array
		srcStart := realDataIdx * rowSize
		srcEnd := srcStart + rowSize

		// We calculate where it should go in the worker's small local buffer
		dstStart := localRowIdx * rowSize
		dstEnd := dstStart + rowSize

		copy(destX.data[dstStart:dstEnd], globalX[srcStart:srcEnd])
	}
}

func main() {
	G := runtime.GOMAXPROCS(runtime.NumCPU())
	fmt.Printf("Running on %d cores (Mini-Batch Mode)\n", runtime.GOMAXPROCS(0))

	// --- 1. Load Data ---
	X_raw, Y_raw, err := data.LoadCSV("mnist_train.csv")
	if err != nil {
		panic("Failed to load data: " + err.Error())
	}
	data.MinMaxNormalize(X_raw)

	inputDim := len(X_raw[0])
	numSamples := len(X_raw)

	// Create the single massive backing array (Zero Copy)
	X_global := NewMatrixFromSlice(len(X_raw), len(X_raw[0]), flatten(X_raw))

	// --- NEW INITIALIZATION SYNTAX ---
	masterNN := NewNetwork(0.1,
		Input(784),
		Dense(64),
		Dense(32),
		Dense(16),
		Dense(10), // Will automatically be treated as Softmax
	)

	// --- 1. Auto-Load Logic ---
	modelFile := "mnist_model.gob"
	if _, err := os.Stat(modelFile); err == nil {
		fmt.Println("Found existing model. Loading weights...")
		if err := masterNN.LoadFromFile(modelFile); err != nil {
			fmt.Printf("Failed to load model: %v. Starting from scratch.\n", err)
		} else {
			fmt.Println("Weights loaded successfully!")
		}
	} else {
		fmt.Println("No existing model found. Starting training from scratch.")
	}

	// --- 2. Graceful Shutdown (Ctrl+C) ---
	// Create a channel to listen for OS signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigChan // Block here until Ctrl+C is pressed
		fmt.Println("\n\nInterrupt received! Saving model...")

		// Save the current state of MasterNN
		if err := masterNN.SaveToFile(modelFile); err != nil {
			fmt.Printf("Error saving model: %v\n", err)
		} else {
			fmt.Printf("Model saved to %s\n", modelFile)
		}

		fmt.Println("Exiting.")
		os.Exit(0)
	}()

	// --- 2. CONFIGURATION ---
	BatchSize := G * 16 // 126 //numSamples-10// Standard mini-batch size

	// Ensure BatchSize is divisible by G for simplicity in this example
	if BatchSize%G != 0 {
		panic("For this example, please ensure BatchSize is divisible by Core Count (G)")
	}

	localBatchSize := BatchSize / G // e.g., 128 / 8 = 16 samples per worker

	// --- 3. PRE-ALLOCATE WORKER MEMORY (Small Buffers) ---
	workers := make([]*NeuralNetwork, G)
	workerGrads := make([][]GradientSet, G)

	for i := 0; i < G; i++ {
		// A. Clone Structure
		workers[i] = masterNN.CloneStructure()

		// B. Allocate Forward Buffers (Z, A)
		// Crucial: We only allocate enough for the LOCAL mini-batch (small!)
		workers[i].InitializeBuffers(localBatchSize)

		// C. Allocate Gradient Buffers (Same size as weights)
		workerGrads[i] = make([]GradientSet, len(masterNN.Layers))
		for l := 0; l < len(masterNN.Layers); l++ {
			rows, cols := masterNN.Layers[l].Weights.rows, masterNN.Layers[l].Weights.cols
			workerGrads[i][l].dW = NewMatrix(rows, cols)
			workerGrads[i][l].db = NewMatrix(1, cols)
		}
	}

	// --- 4. REUSABLE MASTER GRADIENT BUFFER ---
	finalGrads := make([]GradientSet, len(masterNN.Layers))
	for l := 0; l < len(masterNN.Layers); l++ {
		rows, cols := masterNN.Layers[l].Weights.rows, masterNN.Layers[l].Weights.cols
		finalGrads[l].dW = NewMatrix(rows, cols)
		finalGrads[l].db = NewMatrix(1, cols)
	}

	epochs := 20
	start := time.Now()

	// 1. Create the Master Index List
	globalIndices := NewIndexList(numSamples)

	// 2. Pre-allocate Worker Label Buffers (to avoid making new slices inside loop)
	workerLabels := make([][]float64, G)
	for i := 0; i < G; i++ {
		workerLabels[i] = make([]float64, localBatchSize)
	}

	// 3. Re-use performance counters
	workerLosses := make([]float64, G)
	workerAccs := make([]float64, G)

	fmt.Println("Starting Training (Optimized Index Shuffling)...")

	for epoch := 1; epoch <= epochs; epoch++ {

		// --- OPTIMIZATION: Shuffle Indices, NOT Data ---
		ShuffleIndices(globalIndices)

		var totalLoss float64
		var totalAcc float64 // 1. Add accumulator for epoch accuracy
		batchesProcessed := 0

		// --- INNER BATCH LOOP ---
		// We iterate through the data in jumps of BatchSize
		for batchStart := 0; batchStart+BatchSize <= numSamples; batchStart += BatchSize { // skip last batch if incomplete

			var wg sync.WaitGroup
			wg.Add(G)

			// Dispatch Workers
			for i := 0; i < G; i++ {
				go func(id int) {
					defer wg.Done()

					// 1. Determine which "Random Indices" this worker handles
					// The batch indices are contiguous in our shuffled list,
					// but they point to random locations in the data.
					wStart := batchStart + (id * localBatchSize)
					wEnd := wStart + localBatchSize

					// This slice represents the random IDs this worker must process
					myIndices := globalIndices[wStart:wEnd]

					// 2. GATHER Step
					// Copy the data from random global locations into local contiguous memory
					Gather(
						myIndices,
						X_global.data,
						Y_raw,
						inputDim,
						workers[id].InputBuf, // Destination Matrix
						workerLabels[id],     // Destination Label Slice
					)

					// 3. Forward & Backward (Standard)
					workers[id].Forward(workers[id].InputBuf)

					loss, acc := workers[id].ComputeGradients(workers[id].InputBuf, workerLabels[id], workerGrads[id])

					workerLosses[id] = loss
					workerAccs[id] = acc
				}(i)
			}
			wg.Wait()

			// --- AGGREGATION PHASE (Happens every batch!) ---

			// Clear previous
			for l := 0; l < len(finalGrads); l++ {
				finalGrads[l].dW.Reset()
				finalGrads[l].db.Reset()
			}

			// Sum
			for _, gSet := range workerGrads {
				for l := 0; l < len(finalGrads); l++ {
					finalGrads[l].dW.Add(gSet[l].dW)
					finalGrads[l].db.Add(gSet[l].db)
				}
			}

			// Average and Update IMMEDIATELLY
			scale := 1.0 / float64(G)
			masterNN.UpdateWeights(finalGrads, scale)

			// B. Sum up stats from the slice (New code)
			for i := 0; i < G; i++ {
				totalLoss += workerLosses[i] / float64(G)
				totalAcc += workerAccs[i] / float64(G)
			}
			batchesProcessed++
		}

		// 6. Calculate Final Averages
		avgLoss := totalLoss / float64(batchesProcessed)
		avgAcc := (totalAcc / float64(batchesProcessed)) * 100 // Convert to percentage

		fmt.Printf("Epoch %d | Loss: %.4f | Acc: %.2f%% | Time: %v\n", epoch, avgLoss, avgAcc, time.Since(start))
	}

	// ... Inference code ...
	// [After loop ends]
	masterNN.SaveToFile(modelFile) // Save on normal completion
	fmt.Printf("Total Time: %v\n", time.Since(start))

	fmt.Println("Training Complete. Running Inference...")

	// 1. Load Image
	imagePath := "assets/5.jpg"
	pixelData, err := convertJpg1D(imagePath)
	if err != nil {
		panic(err)
	}

	// 2. Normalize (Crucial!)
	// Assuming training data was MinMaxNormalized (0.0 to 1.0).
	// If your JPG reader returns 0-255, you MUST divide by 255.0.
	for i := range pixelData {
		pixelData[i] = pixelData[i] / 255.0
	}

	// 3. Predict
	prediction, confidence := masterNN.Predict(pixelData)

	fmt.Printf("Image: %s\n", imagePath)
	fmt.Printf("Predicted Digit: %d\n", prediction)
	fmt.Printf("Confidence: %.2f%%\n", confidence*100)
}