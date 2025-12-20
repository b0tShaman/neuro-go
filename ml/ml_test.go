package ml

import (
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// --- Global Variables to prevent compiler optimizations ---
var resultMat *Matrix
var resultLoss float64
var resultAcc float64

// --- 1. Benchmarks: Matrix Multiplication ---

func benchmarkMatMul(b *testing.B, size int, method string) {
	m1 := NewMatrix(size, size)
	m2 := NewMatrix(size, size)
	out := NewMatrix(size, size)

	m1.Randomize()
	m2.Randomize()

	b.ResetTimer()

	if method == "Native" {
		for n := 0; n < b.N; n++ {
			// Note: This relies on MatMulGo being present in your main.go
			// If you removed it, comment this block out.
			MatMulGo(m1, m2, out)
		}
	} else {
		for n := 0; n < b.N; n++ {
			// Pass the underlying gonum object (.dense)
			MatMul(m1.dense, m2.dense, out)
		}
	}
	resultMat = out
}

func BenchmarkMatMul_Native_64(b *testing.B)   { benchmarkMatMul(b, 64, "Native") }
func BenchmarkMatMul_Gonum_64(b *testing.B)    { benchmarkMatMul(b, 64, "Gonum") }
func BenchmarkMatMul_Native_256(b *testing.B)  { benchmarkMatMul(b, 256, "Native") }
func BenchmarkMatMul_Gonum_256(b *testing.B)   { benchmarkMatMul(b, 256, "Gonum") }
func BenchmarkMatMul_Native_512(b *testing.B)  { benchmarkMatMul(b, 512, "Native") }
func BenchmarkMatMul_Gonum_512(b *testing.B)   { benchmarkMatMul(b, 512, "Gonum") }
func BenchmarkMatMul_Native_1024(b *testing.B) { benchmarkMatMul(b, 1024, "Native") }
func BenchmarkMatMul_Gonum_1024(b *testing.B)  { benchmarkMatMul(b, 1024, "Gonum") }

// --- 2. Benchmarks: Activation Function Overhead ---

func BenchmarkActivation_FuncPtr(b *testing.B) {
	// 1 Million elements
	m := NewMatrix(1000, 1000)
	m.Randomize()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		// Simulating the overhead of passing a function pointer
		m.ApplyFunc(Relu)
	}
}

func BenchmarkActivation_HardcodedLoop(b *testing.B) {
	// 1 Million elements
	m := NewMatrix(1000, 1000)
	m.Randomize()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		// Simulating the optimized direct loop
		m.ApplyRelu()
	}
}

// --- 3. Benchmarks: Neural Network Operations ---

// setupNetwork prepares a standard MNIST network and buffers
func setupNetwork(batchSize int) (*NeuralNetwork, *Matrix, []float64, []GradientSet) {
	nn := NewNetwork(
		Input(784),
		Dense(64),
		Dense(32),
		Dense(16),
		Dense(10, Activation("softmax")),
	)

	// Important: Initialize internal Z/A buffers
	nn.InitializeBuffers(batchSize)

	// Random Input
	inputData := make([]float64, batchSize*784)
	for i := range inputData {
		inputData[i] = rand.Float64()
	}

	// Create Matrix manually since NewMatrixFromSlice uses local package logic
	// Ensure this matches your Matrix struct definition
	inputMat := &Matrix{
		rows:  batchSize,
		cols:  784,
		data:  inputData,
		dense: mat.NewDense(batchSize, 784, inputData),
	}

	// Random Targets (Labels 0-9)
	targets := make([]float64, batchSize)
	for i := range targets {
		targets[i] = float64(rand.Intn(10))
	}

	// Pre-allocate Gradient buffers (so benchmark doesn't measure allocation)
	grads := make([]GradientSet, len(nn.Layers))
	for l := 0; l < len(nn.Layers); l++ {
		rows, cols := nn.Layers[l].Weights.rows, nn.Layers[l].Weights.cols
		grads[l].dW = NewMatrix(rows, cols)
		grads[l].db = NewMatrix(1, cols)
	}

	return nn, inputMat, targets, grads
}

// Benchmark: Forward Pass Only (Inference Speed)
func benchmarkForward(b *testing.B, batchSize int) {
	nn, input, _, _ := setupNetwork(batchSize)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		nn.Forward(input)
	}
}

func BenchmarkForward_Batch_1(b *testing.B)   { benchmarkForward(b, 1) }
func BenchmarkForward_Batch_64(b *testing.B)  { benchmarkForward(b, 64) }
func BenchmarkForward_Batch_128(b *testing.B) { benchmarkForward(b, 128) }

// Benchmark: Backward Pass Only (Gradient Calculation Cost)
func benchmarkBackprop(b *testing.B, batchSize int) {
	nn, input, targets, grads := setupNetwork(batchSize)

	// Pre-warm the state with one forward pass so Z/A matrices are populated
	nn.Forward(input)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		// We measure ONLY the gradient calculation
		loss, acc := nn.ComputeGradients(input, targets, grads)
		resultLoss = loss
		resultAcc = acc
	}
}

func BenchmarkBackprop_Batch_64(b *testing.B)  { benchmarkBackprop(b, 64) }
func BenchmarkBackprop_Batch_128(b *testing.B) { benchmarkBackprop(b, 128) }

// --- 4. Benchmarks: Optimizer Types (Micro-Benchmark) ---

// setupOptimizerData prepares random gradients to simulate a real update step.
func setupOptimizerData(nn *NeuralNetwork) []GradientSet {
	grads := make([]GradientSet, len(nn.Layers))
	for l := 0; l < len(nn.Layers); l++ {
		rows, cols := nn.Layers[l].Weights.rows, nn.Layers[l].Weights.cols

		// Create gradients
		grads[l].dW = NewMatrix(rows, cols)
		grads[l].db = NewMatrix(1, cols)

		// Fill with random noise
		grads[l].dW.Randomize()
		grads[l].db.Randomize()
	}
	return grads
}

func benchmarkOptimizerUpdate(b *testing.B, optType OptimizerType) {
	// 1. Setup Network (size impacts optimizer cost)
	// Using a larger network to make memory bandwidth matter more
	nn := NewNetwork(
		Input(784),
		Dense(128),
		Dense(128),
		Dense(10, Activation("softmax")),
	)

	// 2. Setup Dummy Gradients
	grads := setupOptimizerData(nn)

	// 3. Create the specific Optimizer
	cfg := TrainingConfig{
		LearningRate: 0.01,
		Optimizer:    optType,
		MomentumMu:   0.9,
		AdamBeta1:    0.9,
		AdamBeta2:    0.999,
		AdamEps:      1e-8,
	}
	optimizer := NewOptimizer(nn, cfg)

	b.ResetTimer()

	// 4. Run the hot loop
	for n := 0; n < b.N; n++ {
		optimizer.Update(nn, grads)
	}
}

func BenchmarkOpt_Micro_SGD(b *testing.B)      { benchmarkOptimizerUpdate(b, OptSGD) }
func BenchmarkOpt_Micro_Momentum(b *testing.B) { benchmarkOptimizerUpdate(b, OptMomentum) }
func BenchmarkOpt_Micro_Adam(b *testing.B)     { benchmarkOptimizerUpdate(b, OptAdam) }

// --- 5. Benchmarks: Full Training Loop with Optimizers (Integrated) ---

func benchmarkFullStepWithOpt(b *testing.B, batchSize int, optType OptimizerType) {
	// 1. Setup
	nn, input, targets, grads := setupNetwork(batchSize)

	// 2. Config & Optimizer
	cfg := TrainingConfig{
		LearningRate: 0.01,
		Optimizer:    optType,
		MomentumMu:   0.9,
		AdamBeta1:    0.9,
		AdamBeta2:    0.999,
		AdamEps:      1e-8,
	}
	optimizer := NewOptimizer(nn, cfg)

	b.ResetTimer()

	for n := 0; n < b.N; n++ {
		// A. Forward
		nn.Forward(input)

		// B. Backward
		nn.ComputeGradients(input, targets, grads)

		// C. Update
		optimizer.Update(nn, grads)
	}
}

// Comparison at Batch Size 64
func BenchmarkTrainStep_SGD_64(b *testing.B)      { benchmarkFullStepWithOpt(b, 64, OptSGD) }
func BenchmarkTrainStep_Momentum_64(b *testing.B) { benchmarkFullStepWithOpt(b, 64, OptMomentum) }
func BenchmarkTrainStep_Adam_64(b *testing.B)     { benchmarkFullStepWithOpt(b, 64, OptAdam) }

// Comparison at Batch Size 256
func BenchmarkTrainStep_SGD_256(b *testing.B)      { benchmarkFullStepWithOpt(b, 256, OptSGD) }
func BenchmarkTrainStep_Momentum_256(b *testing.B) { benchmarkFullStepWithOpt(b, 256, OptMomentum) }
func BenchmarkTrainStep_Adam_256(b *testing.B)     { benchmarkFullStepWithOpt(b, 256, OptAdam) }
