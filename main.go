package main

import (
	"bufio"
	"bytes"
	"encoding/gob"
	"fmt"
	"image"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/b0tShaman/neuro-go/data"

	"golang.org/x/image/draw"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

const (
	ActLinear ActivationType = iota
	ActRelu
	ActSigmoid
	ActSoftmax
	ActEmbedding
)

const (
	OptSGD      OptimizerType = "sgd"
	OptMomentum OptimizerType = "momentum"
	OptAdam     OptimizerType = "adam"
)

var activationMap = map[string]ActivationType{
	"linear":  ActLinear,
	"sigmoid": ActSigmoid,
	"relu":    ActRelu,
	"softmax": ActSoftmax,
}

// Default settings generally recommended for Adam
var DefaultAdamConfig = AdamConfig{
	Beta1:        0.9,
	Beta2:        0.999,
	Epsilon:      1e-8,
	LearningRate: 0.001,
}

// -------- TYPE DEFINITIONS -------- //
type ActivationType int
type OptimizerType string
type LayerOption func(*LayerConfig)

// LayerConfig holds the blueprint for a layer
type LayerConfig struct {
	Neurons    int
	IsInput    bool
	Activation ActivationType
	// Fields for Embeddings
	VocabSize int
	EmbedDim  int
}

type LayerState struct {
	// Moving averages for Weights
	mW *Matrix
	vW *Matrix

	// Moving averages for Biases
	mB *Matrix
	vB *Matrix
}

type Layer struct {
	Weights *Matrix
	Biases  *Matrix

	// Forward State
	Z *Matrix
	A *Matrix

	// Backward State
	dZ      *Matrix
	ActType ActivationType
}

type NeuralNetwork struct {
	Layers       []*Layer
	LearningRate float64
	InputT       *Matrix
	InputBuf     *Matrix
}

// GradientSet holds the calculated gradients for one layer
type GradientSet struct {
	dW *Matrix
	db *Matrix
}

type Optimizer interface {
	Update(nw *NeuralNetwork, grads []GradientSet)
}

type AdamConfig struct {
	Beta1        float64
	Beta2        float64
	Epsilon      float64
	LearningRate float64
}

type AdamOptimizer struct {
	cfg         AdamConfig
	layerStates []LayerState
	timeStep    int // 't' in the Adam paper, tracks number of updates
}

type SGDOptimizer struct {
	LearningRate float64
}

type MomentumOptimizer struct {
	LearningRate float64
	Mu           float64
	velocities   [][]GradientSet // State: vW, vB per layer
}

type TrainingConfig struct {
	Epochs       int
	BatchSize    int
	LearningRate float64
	ModelPath    string
	NumWorkers   int

	// Optimizer Selection
	Optimizer OptimizerType

	// Optimizer Hyperparameters (Zero values will use defaults)
	MomentumMu float64 // For Momentum (usually 0.9)
	AdamBeta1  float64 // For Adam (usually 0.9)
	AdamBeta2  float64 // For Adam (usually 0.999)
	AdamEps    float64 // For Adam (usually 1e-8)
}

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

// Neural Network Builder
func NewNetwork(configs ...LayerConfig) *NeuralNetwork {
	if len(configs) < 2 {
		panic("Network must have at least Input and one Output layer")
	}
	if !configs[0].IsInput {
		panic("First layer must be Input()")
	}

	nn := &NeuralNetwork{}
	prevOutputSize := configs[0].Neurons

	for i := 1; i < len(configs); i++ {
		cfg := configs[i]

		var weightsRows, weightsCols int

		// Special Handling for Embedding Layer
		if cfg.Activation == ActEmbedding {
			if cfg.VocabSize == 0 {
				panic("VocabSize must be set for Embedding layer")
			}

			// 2. Mandatory Position Check
			// Since i=0 is Input, i=1 is the very first hidden layer.
			// This ensures: Input -> Embedding -> ...
			if i != 1 {
				panic("Embedding layer must be the second layer immediately after Input")
			}

			// Weights are (VocabSize x EmbedDim)
			weightsRows = cfg.VocabSize
			weightsCols = cfg.EmbedDim

			// The output of this layer is flattened: ContextLen * EmbedDim
			if cfg.Neurons == 0 {
				cfg.Neurons = prevOutputSize * cfg.EmbedDim
			}
		} else {
			// Standard Dense Layer
			weightsRows = prevOutputSize
			weightsCols = cfg.Neurons
		}

		layer := &Layer{
			Weights: NewMatrix(weightsRows, weightsCols),
			Biases:  NewMatrix(1, cfg.Neurons), // Biases usually unused in Embedding, but kept for structural consistency
			ActType: cfg.Activation,
		}

		// Initialize Weights
		if cfg.Activation == ActEmbedding {
			// Use Standard Random (or tailored Embedding init)
			layer.Weights.Randomize()
		} else {
			// Use Xavier/Glorot for Dense layers
			layer.Weights.RandomizeXavier()
		}

		nn.Layers = append(nn.Layers, layer)
		prevOutputSize = cfg.Neurons
	}

	return nn
}

func NewOptimizer(nw *NeuralNetwork, cfg TrainingConfig) Optimizer {
	switch cfg.Optimizer {
	case OptAdam:
		// Set defaults if 0
		beta1 := cfg.AdamBeta1
		if beta1 == 0 {
			beta1 = 0.9
		}
		beta2 := cfg.AdamBeta2
		if beta2 == 0 {
			beta2 = 0.999
		}
		eps := cfg.AdamEps
		if eps == 0 {
			eps = 1e-8
		}

		adamCfg := AdamConfig{
			Beta1:        beta1,
			Beta2:        beta2,
			Epsilon:      eps,
			LearningRate: cfg.LearningRate,
		}
		return NewAdamOptimizer(nw, adamCfg)

	case OptMomentum:
		return NewMomentumOptimizer(nw, cfg.LearningRate, cfg.MomentumMu)

	case OptSGD:
		return &SGDOptimizer{LearningRate: cfg.LearningRate}

	default:
		return &SGDOptimizer{LearningRate: cfg.LearningRate}
	}
}

func NewAdamOptimizer(nw *NeuralNetwork, cfg AdamConfig) *AdamOptimizer {
	opt := &AdamOptimizer{
		cfg:      cfg,
		timeStep: 0,
	}

	// Initialize zero-matrices for every layer's parameters
	for _, layer := range nw.Layers {
		state := LayerState{
			mW: NewMatrix(layer.Weights.rows, layer.Weights.cols),
			vW: NewMatrix(layer.Weights.rows, layer.Weights.cols),
			mB: NewMatrix(layer.Biases.rows, layer.Biases.cols),
			vB: NewMatrix(layer.Biases.rows, layer.Biases.cols),
		}
		// Ensure they start at zero (NewMatrix usually does this, but being explicit helps)
		state.mW.Reset()
		state.vW.Reset()
		state.mB.Reset()
		state.vB.Reset()

		opt.layerStates = append(opt.layerStates, state)
	}

	return opt
}

func NewMomentumOptimizer(nw *NeuralNetwork, lr, mu float64) *MomentumOptimizer {
	if mu == 0 {
		mu = 0.9
	} // Default

	opt := &MomentumOptimizer{
		LearningRate: lr,
		Mu:           mu,
		velocities:   make([][]GradientSet, len(nw.Layers)),
	}

	for i, layer := range nw.Layers {
		opt.velocities[i] = make([]GradientSet, 1)
		opt.velocities[i][0].dW = NewMatrix(layer.Weights.rows, layer.Weights.cols)
		opt.velocities[i][0].db = NewMatrix(layer.Biases.rows, layer.Biases.cols)
	}
	return opt
}

// ------- LAYER CONFIG HELPERS ------- //
// Input defines the entry point dimensions
func Input(size int) LayerConfig {
	return LayerConfig{
		Neurons:    size,
		IsInput:    true,
		Activation: ActLinear,
	}
}

// Dense defines a fully connected layer.
func Dense(size int, opts ...LayerOption) LayerConfig {
	d := LayerConfig{
		Neurons:    size,
		IsInput:    false,
		Activation: ActRelu, // Default for hidden layers
	}

	for _, opt := range opts {
		opt(&d)
	}
	return d
}

func Activation(activation string) LayerOption {
	return func(lc *LayerConfig) {
		act, exists := activationMap[activation]
		if !exists {
			panic("Unknown activation: " + activation)
		}
		lc.Activation = act
	}
}

// Embedding Layer Constructor
// OutputDim is optional here if you want to override the default flattening
func Embedding(embedDim int, opts ...LayerOption) LayerConfig {
	cfg := LayerConfig{
		Activation: ActEmbedding,
		EmbedDim:   embedDim,
		IsInput:    false,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	// If Neurons (Output Size) wasn't manually set, it is calculated in NewNetwork
	// based on ContextLen * EmbedDim
	return cfg
}

func VocabSize(size int) LayerOption {
	return func(lc *LayerConfig) {
		lc.VocabSize = size
	}
}

func OutputDim(size int) LayerOption {
	return func(lc *LayerConfig) {
		lc.Neurons = size
	}
}

// -------- MAIN -------- //
func main() {
	// Hardware Setup
	G := runtime.GOMAXPROCS(runtime.NumCPU())
	modelFile := "assets/model.gob"

	// 1. Load Data
	fmt.Println("Loading dataset...")

	// Image Data
	// X_raw, Y_raw, err := data.LoadCSV("assets/mnist_train.csv")
	// if err != nil {
	//  panic("Failed to load data: " + err.Error())
	// }

	// Text Data
	embedDim := 4
	contextLen := 5
	freqThreshold := 1 // Replace words with freq < threshold with <unk>

	var X_raw [][]float64
	var Y any
	var err error

	outputDim := contextLen * embedDim // Output dimension for Embedding layer for non-sequential model(n-gram)

	vocabSize, wordToID, idToWord := data.PreprocessNGramData(
		contextLen,
		freqThreshold,
		"assets/input.txt",
		"assets/",
	)

	X_raw, Y, err = data.LoadCSV("assets/dataset.csv")
	Y_raw := Y.([]float64)

	if err != nil {
		fmt.Println("Error in Preprocess:", err)
		return
	}

	fmt.Println("✅ Vocabulary size:", vocabSize)

	inputDim := len(X_raw[0])
	numSamples := len(X_raw)

	fmt.Printf("Loaded dataset: %d samples, %d input features\n", numSamples, inputDim)

	// Create Global Matrix (Zero Copy)
	X_global := NewMatrixFromSlice(len(X_raw), len(X_raw[0]), flatten(X_raw))

	// 2. Initialize Network
	nw := NewNetwork(
		Input(inputDim),
		Embedding(embedDim, VocabSize(vocabSize), OutputDim(outputDim)),
		Dense(128),
		Dense(128),
		Dense(vocabSize, Activation("softmax")),
	)

	// Auto-Load weights if they exist
	if _, err := os.Stat(modelFile); err == nil {
		fmt.Println("Found existing model. Loading weights...")
		err := nw.LoadFromFile(modelFile)
		if err != nil {
			// panic(fmt.Sprintf("Cannot load model: %v", err))
			// OPTION B: Log warning and start fresh (safer for dev)
			fmt.Printf("⚠️ Model mismatch (%v). Starting training from scratch.\n", err)
		}
	}

	// 3. Configure & Train
	config := TrainingConfig{
		Epochs:       150,
		BatchSize:    G * 16,
		LearningRate: 0.001, // Adam default is 0.001, SGD can be higher
		ModelPath:    modelFile,
		NumWorkers:   G,
		Optimizer:    OptAdam,
		MomentumMu:   0.9, // Only used if OptMomentum is selected
	}

	fmt.Printf("Running on %d cores (Mini-Batch = %d)\n\n", runtime.GOMAXPROCS(0), config.BatchSize)

	// Text Training and Inference
	nw.TrainTxt(X_global, Y_raw, config)
	nw.InferenceTxt(wordToID, idToWord)

	// Img Training and Inference
	// nw.TrainImg(X_global, Y_raw, config)
	// nw.InferenceImg("assets/5.jpg")
}

func (nw *NeuralNetwork) TrainTxt(X_ids *Matrix, Y_ids []float64, cfg TrainingConfig) {
	fmt.Printf("TrainingConfig: %+v\n", cfg)

	if cfg.BatchSize%cfg.NumWorkers != 0 {
		panic("BatchSize must be divisible by NumWorkers")
	}

	optimizer := NewOptimizer(nw, cfg)
	localBatchSize := cfg.BatchSize / cfg.NumWorkers
	inputCtxLen := X_ids.cols // This is contextLen
	numSamples := X_ids.rows

	// 1. PRE-ALLOCATE WORKER MEMORY
	workers := make([]*NeuralNetwork, cfg.NumWorkers)
	workerGrads := make([][]GradientSet, cfg.NumWorkers)

	fmt.Printf("Initializing %d workers (Worker Batch: %d)\n", cfg.NumWorkers, localBatchSize)

	for i := 0; i < cfg.NumWorkers; i++ {
		workers[i] = nw.CloneStructure()
		// Important: InitializeBuffers determines size based on Weights.
		// For Embedding layer, InputBuf must match [Batch, ContextLen].
		workers[i].InitializeBuffers(localBatchSize)

		workerGrads[i] = make([]GradientSet, len(nw.Layers))
		for l := 0; l < len(nw.Layers); l++ {
			// [FIX] Source dimensions separately for Weights and Biases
			wRows, wCols := nw.Layers[l].Weights.rows, nw.Layers[l].Weights.cols
			bRows, bCols := nw.Layers[l].Biases.rows, nw.Layers[l].Biases.cols

			workerGrads[i][l].dW = NewMatrix(wRows, wCols)
			workerGrads[i][l].db = NewMatrix(bRows, bCols) // Use bCols, not wCols
		}
	}

	// 2. MASTER GRADIENT BUFFER
	finalGrads := make([]GradientSet, len(nw.Layers))
	for l := 0; l < len(nw.Layers); l++ {
		// [FIX] Source dimensions separately
		wRows, wCols := nw.Layers[l].Weights.rows, nw.Layers[l].Weights.cols
		bRows, bCols := nw.Layers[l].Biases.rows, nw.Layers[l].Biases.cols

		finalGrads[l].dW = NewMatrix(wRows, wCols)
		finalGrads[l].db = NewMatrix(bRows, bCols) // Use bCols, not wCols
	}

	globalIndices := NewIndexList(numSamples)

	// Label buffers
	workerLabels := make([][]float64, cfg.NumWorkers)
	for i := 0; i < cfg.NumWorkers; i++ {
		workerLabels[i] = make([]float64, localBatchSize)
	}

	workerLosses := make([]float64, cfg.NumWorkers)
	workerAccs := make([]float64, cfg.NumWorkers)

	// Signal Handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigChan
		fmt.Println("\nInterrupt! Saving model...")
		nw.SaveToFile(cfg.ModelPath)
		os.Exit(0)
	}()

	start := time.Now()
	fmt.Println("Starting Training...")

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		ShuffleIndices(globalIndices)

		var totalLoss, totalAcc float64
		batchesProcessed := 0

		for batchStart := 0; batchStart+cfg.BatchSize <= numSamples; batchStart += cfg.BatchSize {
			var wg sync.WaitGroup
			wg.Add(cfg.NumWorkers)

			// Dispatch Workers
			for i := 0; i < cfg.NumWorkers; i++ {
				go func(id int) {
					defer wg.Done()
					wStart := batchStart + (id * localBatchSize)
					wEnd := wStart + localBatchSize
					myIndices := globalIndices[wStart:wEnd]

					// Gather works for Indices too (copies floats)
					Gather(myIndices, X_ids.data, Y_ids, inputCtxLen, workers[id].InputBuf, workerLabels[id])

					workers[id].Forward(workers[id].InputBuf)
					loss, acc := workers[id].ComputeGradients(workers[id].InputBuf, workerLabels[id], workerGrads[id])

					workerLosses[id] = loss
					workerAccs[id] = acc
				}(i)
			}
			wg.Wait()

			// Aggregate Gradients
			for l := range finalGrads {
				finalDW := finalGrads[l].dW
				finalDB := finalGrads[l].db
				copy(finalDW.data, workerGrads[0][l].dW.data)
				copy(finalDB.data, workerGrads[0][l].db.data)

				for w := 1; w < cfg.NumWorkers; w++ {
					floats.Add(finalDW.data, workerGrads[w][l].dW.data)
					floats.Add(finalDB.data, workerGrads[w][l].db.data)
				}
			}

			// Apply Scale & Update
			scale := 1.0 / float64(cfg.NumWorkers)
			for l := range finalGrads {
				floats.Scale(scale, finalGrads[l].dW.data)
				floats.Scale(scale, finalGrads[l].db.data)
			}

			optimizer.Update(nw, finalGrads)

			for i := range cfg.NumWorkers {
				totalLoss += workerLosses[i] / float64(cfg.NumWorkers)
				totalAcc += workerAccs[i] / float64(cfg.NumWorkers)
			}
			batchesProcessed++
		}

		avgLoss := totalLoss / float64(batchesProcessed)
		avgAcc := (totalAcc / float64(batchesProcessed)) * 100
		if epoch%10 == 0 || epoch == 1 {
			fmt.Printf("Epoch %d | Loss: %.4f | Acc: %.2f%% | Time: %v\n", epoch, avgLoss, avgAcc, time.Since(start))
		}
	}

	nw.SaveToFile(cfg.ModelPath)
	fmt.Printf("Training Complete. Total Time: %v\n\n", time.Since(start))
}

func (nw *NeuralNetwork) InferenceTxt(wordToID map[string]int, idToWord []string) {
	// 1. Setup & Context Calculation
	reader := bufio.NewReader(os.Stdin)

	// Ensure buffers are allocated (BatchSize = 1)
	if nw.Layers[0].A == nil {
		nw.InitializeBuffers(1)
	}

	// Determine Context Length from Architecture
	firstLayer := nw.Layers[0]
	var contextLen int
	if firstLayer.ActType == ActEmbedding {
		// Embeddings: ContextLen = OutputSize / EmbedDim
		contextLen = firstLayer.Biases.cols / firstLayer.Weights.cols
	} else {
		contextLen = firstLayer.Weights.rows
	}

	// Constants for Tokenization
	const PAD = "<pad>"
	const UNK = "<unk>"
	const EOS = "eos" // Stop generation at this token

	// 2. Interactive Loop
	for {
		fmt.Print("\nInput: ")
		line, _ := reader.ReadString('\n')
		line = strings.TrimSpace(strings.ToLower(line))

		if line == "exit" || line == "quit" {
			break
		}
		if line == "" {
			continue
		}

		// --- PRE-PROCESSING ---
		words := strings.Fields(line)

		// Pad or Trim to fit Context Length
		if len(words) < contextLen {
			// Pad logic: Prepend PAD tokens
			pads := make([]string, contextLen-len(words))
			for i := range pads {
				pads[i] = PAD
			}
			words = append(pads, words...)
		} else if len(words) > contextLen {
			// Trim logic: Keep last N words
			words = words[len(words)-contextLen:]
		}

		// Convert Words -> IDs (Float64 Window)
		window := make([]float64, contextLen)
		for i := 0; i < contextLen; i++ {
			w := words[i]
			id, exists := wordToID[w]
			if !exists {
				id = wordToID[UNK]
			}
			window[i] = float64(id)
		}

		// --- GENERATION LOOP ---

		maxTokens := 50 // Safety limit

		for i := 0; i < maxTokens; i++ {
			// A. Create Input Matrix (Zero Copy)
			inputMat := NewMatrixFromSlice(1, contextLen, window)

			// B. Forward Pass
			nw.Forward(inputMat)

			// C. Get Probabilities
			lastLayer := nw.Layers[len(nw.Layers)-1]
			probs := lastLayer.A.data

			// D. Argmax (Greedy Decoding)
			bestID := -1
			maxProb := -1.0
			for id, p := range probs {
				if p > maxProb {
					maxProb = p
					bestID = id
				}
			}

			// E. Decode & Print
			if bestID >= len(idToWord) {
				break
			} // Safety check

			word := idToWord[bestID]
			fmt.Print(word + " ")
			time.Sleep(100 * time.Millisecond)

			// F. Stop Conditions
			if word == EOS || word == "." {
				break
			}

			// G. Shift Window (Sliding Context)
			// [A, B, C] -> [B, C, NewID]
			copy(window[0:], window[1:])
			window[contextLen-1] = float64(bestID)
		}
		fmt.Println() // Newline after response
	}
}

// -------- NEURAL NETWORK METHODS -------- //
func (nw *NeuralNetwork) TrainImg(X_global *Matrix, Y_raw []float64, cfg TrainingConfig) {
	fmt.Printf("TrainingConfig: %+v\n", cfg)

	if cfg.BatchSize%cfg.NumWorkers != 0 {
		panic("BatchSize must be divisible by NumWorkers")
	}

	for i := range X_global.data {
		X_global.data[i] /= 255.0
	}

	// Create the chosen optimizer
	optimizer := NewOptimizer(nw, cfg)

	localBatchSize := cfg.BatchSize / cfg.NumWorkers
	inputDim := X_global.cols
	numSamples := X_global.rows

	// 1. PRE-ALLOCATE WORKER MEMORY
	workers := make([]*NeuralNetwork, cfg.NumWorkers)
	workerGrads := make([][]GradientSet, cfg.NumWorkers)

	fmt.Printf("Initializing %d workers (Worker Batch: %d)\n", cfg.NumWorkers, localBatchSize)

	for i := 0; i < cfg.NumWorkers; i++ {
		workers[i] = nw.CloneStructure()
		workers[i].InitializeBuffers(localBatchSize)

		workerGrads[i] = make([]GradientSet, len(nw.Layers))
		for l := 0; l < len(nw.Layers); l++ {
			rows, cols := nw.Layers[l].Weights.rows, nw.Layers[l].Weights.cols
			workerGrads[i][l].dW = NewMatrix(rows, cols)
			workerGrads[i][l].db = NewMatrix(1, cols)
		}
	}

	// 2. MASTER GRADIENT BUFFER
	finalGrads := make([]GradientSet, len(nw.Layers))
	for l := 0; l < len(nw.Layers); l++ {
		rows, cols := nw.Layers[l].Weights.rows, nw.Layers[l].Weights.cols
		finalGrads[l].dW = NewMatrix(rows, cols)
		finalGrads[l].db = NewMatrix(1, cols)
	}

	// 3. AUXILIARY BUFFERS
	globalIndices := NewIndexList(numSamples)

	// Label buffers (avoid alloc inside loop)
	workerLabels := make([][]float64, cfg.NumWorkers)
	for i := 0; i < cfg.NumWorkers; i++ {
		workerLabels[i] = make([]float64, localBatchSize)
	}

	// Stats buffers
	workerLosses := make([]float64, cfg.NumWorkers)
	workerAccs := make([]float64, cfg.NumWorkers)

	// 4. SIGNAL HANDLING (Ctrl+C)
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigChan
		fmt.Println("\n\nInterrupt received! Saving model...")
		nw.SaveToFile(cfg.ModelPath)
		os.Exit(0)
	}()

	// 5. TRAINING LOOP
	start := time.Now()
	fmt.Println("Starting Training...")

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		ShuffleIndices(globalIndices)

		var totalLoss, totalAcc float64
		batchesProcessed := 0

		for batchStart := 0; batchStart+cfg.BatchSize <= numSamples; batchStart += cfg.BatchSize {
			var wg sync.WaitGroup
			wg.Add(cfg.NumWorkers)

			// A. Dispatch Workers
			for i := 0; i < cfg.NumWorkers; i++ {
				go func(id int) {
					defer wg.Done()

					// Calc indices
					wStart := batchStart + (id * localBatchSize)
					wEnd := wStart + localBatchSize
					myIndices := globalIndices[wStart:wEnd]

					// Gather Data
					Gather(myIndices, X_global.data, Y_raw, inputDim, workers[id].InputBuf, workerLabels[id])

					// Compute
					workers[id].Forward(workers[id].InputBuf)
					loss, acc := workers[id].ComputeGradients(workers[id].InputBuf, workerLabels[id], workerGrads[id])

					workerLosses[id] = loss
					workerAccs[id] = acc
				}(i)
			}
			wg.Wait()

			// B. Aggregate Gradients (Optimized Copy+Add)
			for l := range finalGrads {
				finalDW := finalGrads[l].dW
				finalDB := finalGrads[l].db

				// 1. Copy Worker 0 (Fastest init)
				copy(finalDW.data, workerGrads[0][l].dW.data)
				copy(finalDB.data, workerGrads[0][l].db.data)

				// 2. Add remaining workers
				for w := 1; w < cfg.NumWorkers; w++ {
					floats.Add(finalDW.data, workerGrads[w][l].dW.data)
					floats.Add(finalDB.data, workerGrads[w][l].db.data)
				}
			}

			// C. Update Weights
			// scale := 1.0 / float64(cfg.NumWorkers)
			// nw.UpdateWeights(finalGrads, scale)
			scale := 1.0 / float64(cfg.NumWorkers)

			for l := range finalGrads {
				// Apply scaling directly to the gradient matrices before passing to Adam
				floats.Scale(scale, finalGrads[l].dW.data)
				floats.Scale(scale, finalGrads[l].db.data)
			}

			optimizer.Update(nw, finalGrads)

			// D. Aggregate Stats
			for i := range cfg.NumWorkers {
				totalLoss += workerLosses[i] / float64(cfg.NumWorkers)
				totalAcc += workerAccs[i] / float64(cfg.NumWorkers)
			}
			batchesProcessed++
		}

		avgLoss := totalLoss / float64(batchesProcessed)
		avgAcc := (totalAcc / float64(batchesProcessed)) * 100
		fmt.Printf("Epoch %d | Loss: %.4f | Acc: %.2f%% | Time: %v\n", epoch, avgLoss, avgAcc, time.Since(start))
	}

	// Save final model
	nw.SaveToFile(cfg.ModelPath)
	fmt.Printf("Training Complete. Total Time: %v\n\n", time.Since(start))
}

func (nw *NeuralNetwork) InferenceImg(imagePath string) {
	fmt.Printf("Running Inference on: %s\n", imagePath)

	// 1. Load & Convert
	pixelData, err := convertJpg1D(imagePath, 28, 28)
	if err != nil {
		fmt.Printf("Error loading image: %v\n", err)
		return
	}

	// 2. Normalize (0-255 -> 0.0-1.0)
	for i := range pixelData {
		pixelData[i] = pixelData[i] / 255.0
	}

	// 3. Predict
	prediction, confidence := nw.Predict(pixelData)

	fmt.Printf("Predicted Digit: %d\n", prediction)
	fmt.Printf("Confidence: %.2f%%\n", confidence*100)
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

func (nw *NeuralNetwork) InitializeBuffers(batchSize int) {
	inputDim := nw.Layers[0].Weights.rows
	if nw.Layers[0].ActType == ActEmbedding {
		inputDim = nw.Layers[0].Biases.cols / nw.Layers[0].Weights.cols
	}

	data := make([]float64, batchSize*inputDim)
	nw.InputBuf = &Matrix{
		rows:  batchSize,
		cols:  inputDim,
		data:  data,
		dense: mat.NewDense(batchSize, inputDim, data),
	}
	nw.InputT = NewMatrix(inputDim, batchSize)

	for _, layer := range nw.Layers {
		var outputDim int
		if layer.ActType == ActEmbedding {
			outputDim = layer.Biases.cols
		} else {
			outputDim = layer.Weights.cols
		}

		layer.Z = NewMatrix(batchSize, outputDim)
		layer.A = NewMatrix(batchSize, outputDim)
		layer.dZ = NewMatrix(batchSize, outputDim)
	}
}

func (nw *NeuralNetwork) CloneStructure() *NeuralNetwork {
	newNN := &NeuralNetwork{
		LearningRate: nw.LearningRate,
		Layers:       make([]*Layer, len(nw.Layers)),
	}
	for i, l := range nw.Layers {
		newNN.Layers[i] = &Layer{
			Weights: l.Weights,
			Biases:  l.Biases,
			ActType: l.ActType, // Copy activation type
		}
	}
	return newNN
}

func (nw *NeuralNetwork) Forward(input *Matrix) {
	activation := input
	for _, layer := range nw.Layers {

		if layer.ActType == ActEmbedding {
			// --- EMBEDDING FORWARD (Lookup & Flatten) ---
			// Input: [BatchSize, ContextLen] (Indices stored as floats)
			// Weights: [VocabSize, EmbedDim]
			// Output: [BatchSize, ContextLen * EmbedDim]

			batchSize := input.rows
			contextLen := input.cols
			embedDim := layer.Weights.cols

			expectedRows := batchSize
			expectedCols := contextLen * embedDim

			if layer.A.rows != expectedRows || layer.A.cols != expectedCols {
				panic(fmt.Sprintf("Embedding Buffer Error: Got [%d, %d], Want [%d, %d]. Did you call InitializeBuffers?",
					layer.A.rows, layer.A.cols, expectedRows, expectedCols))
			}

			// Flattened Lookup
			// We map every input ID to its row in Weights and copy it to A
			wData := layer.Weights.data
			outData := layer.A.data
			inData := input.data

			for b := 0; b < batchSize; b++ {
				for t := 0; t < contextLen; t++ {
					// Get Word ID
					wordID := int(inData[b*contextLen+t])

					// Bound check
					if wordID >= layer.Weights.rows || wordID < 0 {
						panic(fmt.Sprintf("Word ID %d out of bounds (Vocab: %d)", wordID, layer.Weights.rows))
					}

					// Copy Embedding Vector
					// Source: Weights row [wordID]
					// Dest: A row [b], sequence position [t]

					srcStart := wordID * embedDim
					dstStart := (b * contextLen * embedDim) + (t * embedDim)

					copy(outData[dstStart:dstStart+embedDim], wData[srcStart:srcStart+embedDim])
				}
			}
			// Embedding has no bias or activation function, strictly lookup
			activation = layer.A

		} else {
			// --- STANDARD DENSE FORWARD ---
			MatMul(activation.dense, layer.Weights.dense, layer.Z)
			layer.Z.AddVector(layer.Biases)
			copy(layer.A.data, layer.Z.data)

			switch layer.ActType {
			case ActSoftmax:
				SoftmaxRow(layer.A)
			case ActRelu:
				layer.A.ApplyRelu()
			case ActSigmoid:
				layer.A.ApplySigmoid()
			case ActLinear:
			default:
				panic("Unknown activation type")
			}
			activation = layer.A
		}
	}
}

// Modify ComputeGradients to handle Embedding Backward Pass
func (nw *NeuralNetwork) ComputeGradients(input *Matrix, Y []float64, grads []GradientSet) (float64, float64) {
	loss, acc := nw.ComputeLossAndAccuracy(Y)

	batchSize := float64(input.rows)
	scale := 1.0 / batchSize

	lastLayerIdx := len(nw.Layers) - 1
	lastLayer := nw.Layers[lastLayerIdx]

	// 1. Output Error (Softmax + CrossEntropy)
	// (Same as before...)
	if lastLayer.ActType == ActSoftmax {
		copy(lastLayer.dZ.data, lastLayer.A.data)
		for i, classLabel := range Y {
			idx := i*lastLayer.dZ.cols + int(classLabel)
			lastLayer.dZ.data[idx] -= 1.0
		}
	} else {
		panic("Only Softmax output layer is currently supported")
	}

	// 2. Backprop Loop
	for i := lastLayerIdx; i >= 0; i-- {
		layer := nw.Layers[i]

		// Determine A_prev
		var A_prev_dense mat.Matrix
		var A_prev_matrix *Matrix // Keep raw ref for embedding

		if i == 0 {
			A_prev_dense = input.dense
			A_prev_matrix = input
		} else {
			A_prev_dense = nw.Layers[i-1].A.dense
			A_prev_matrix = nw.Layers[i-1].A
		}

		if layer.ActType == ActEmbedding {
			// --- EMBEDDING BACKWARD ---
			// We need to map dZ back to dW rows based on Input indices
			// layer.dZ shape: [Batch, ContextLen * EmbedDim]
			// grads[i].dW shape: [Vocab, EmbedDim]

			grads[i].dW.Reset()

			embedDim := layer.Weights.cols
			contextLen := A_prev_matrix.cols // Input is [Batch, ContextLen]

			// dW is already zeroed by the worker logic before this function
			dwData := grads[i].dW.data
			dzData := layer.dZ.data
			inData := A_prev_matrix.data

			for b := 0; b < int(batchSize); b++ {
				for t := range contextLen {
					wordID := int(inData[b*contextLen+t])
					// Safety Check (Optional but recommended)
					if wordID < 0 || wordID >= layer.Weights.rows {
						continue
					}
					// Source Gradients (dZ)
					dZStart := (b * contextLen * embedDim) + (t * embedDim)

					// Dest Gradients (dW row)
					dWStart := wordID * embedDim

					// Accumulate gradients
					// Note: Multiple inputs in batch might point to same WordID, so we ADD.
					for k := range embedDim {
						dwData[dWStart+k] += dzData[dZStart+k]
					}
				}
			}

			// Apply Scale to dW
			// grads[i].dW.ApplyFunc(func(v float64) float64 { return v * scale })

			floats.Scale(scale, grads[i].dW.data) // much faster
			// No Bias update for Embeddings

		} else {
			// --- STANDARD DENSE BACKWARD ---
			if i > 0 {
				MatMul(A_prev_dense.T(), layer.dZ.dense, grads[i].dW)
			} else {
				// Special case: If Layer 0 is Dense, Input is standard float vector
				MatMul(input.dense.T(), layer.dZ.dense, grads[i].dW)
			}

			// Calc db
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

			// grads[i].dW.ApplyFunc(func(v float64) float64 { return v * scale })
			// grads[i].db.ApplyFunc(func(v float64) float64 { return v * scale })

			floats.Scale(scale, grads[i].dW.data) // much faster
			floats.Scale(scale, grads[i].db.data)

			// --- CALC dZ_prev ---
			if i > 0 {
				prevLayer := nw.Layers[i-1]

				// Standard dense backprop to previous layer
				MatMul(layer.dZ.dense, layer.Weights.dense.T(), prevLayer.dZ)

				// Apply Activation Derivative of Previous Layer
				// (Only if previous layer is NOT embedding, as Embedding lookup has no derivative)
				if prevLayer.ActType != ActEmbedding {
					zData := prevLayer.Z.data
					dZPrevData := prevLayer.dZ.data
					for k := range dZPrevData {
						switch prevLayer.ActType {
						case ActRelu:
							if zData[k] <= 0 {
								dZPrevData[k] = 0
							}
						case ActSigmoid:
							a := prevLayer.A.data[k]
							dZPrevData[k] *= a * (1.0 - a)
						}
					}
				}
			}
		}
	}
	return loss, acc
}

// SaveToFile saves the neural network weights and biases to a file.
// Save/Load logic needs to store the ActivationType now
func (nw *NeuralNetwork) SaveToFile(filename string) error {
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

	ld := make([]LayerData, len(nw.Layers))
	for i, l := range nw.Layers {
		ld[i] = LayerData{Weights: l.Weights, Biases: l.Biases, ActType: l.ActType}
	}

	return encoder.Encode(NetworkData{LayerDatas: ld, LearningRate: nw.LearningRate})
}

func (nw *NeuralNetwork) LoadFromFile(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	decoder := gob.NewDecoder(file)

	// Temporary struct to hold the loaded data
	type LayerData struct {
		Weights *Matrix
		Biases  *Matrix
		ActType ActivationType
	}
	type NetworkData struct {
		LayerDatas   []LayerData
		LearningRate float64
	}

	var loadedData NetworkData
	if err := decoder.Decode(&loadedData); err != nil {
		return fmt.Errorf("failed to decode gob file: %v", err)
	}

	// --- VALIDATION STEP ---

	// 1. Check Layer Count
	if len(nw.Layers) != len(loadedData.LayerDatas) {
		return fmt.Errorf("architecture mismatch: current network has %d layers, model file has %d",
			len(nw.Layers), len(loadedData.LayerDatas))
	}

	// 2. Check Dimensions & Types for each layer
	for i, currLayer := range nw.Layers {
		loadedLayer := loadedData.LayerDatas[i]

		// Check Activation Type
		if currLayer.ActType != loadedLayer.ActType {
			return fmt.Errorf("layer %d mismatch: expected activation %v, got %v",
				i, currLayer.ActType, loadedLayer.ActType)
		}

		// Check Weight Dimensions
		if currLayer.Weights.rows != loadedLayer.Weights.rows ||
			currLayer.Weights.cols != loadedLayer.Weights.cols {
			return fmt.Errorf("layer %d weight shape mismatch: expected [%d, %d], got [%d, %d]",
				i,
				currLayer.Weights.rows, currLayer.Weights.cols,
				loadedLayer.Weights.rows, loadedLayer.Weights.cols,
			)
		}

		// Check Bias Dimensions
		if currLayer.Biases.rows != loadedLayer.Biases.rows ||
			currLayer.Biases.cols != loadedLayer.Biases.cols {
			return fmt.Errorf("layer %d bias shape mismatch: expected [%d, %d], got [%d, %d]",
				i,
				currLayer.Biases.rows, currLayer.Biases.cols,
				loadedLayer.Biases.rows, loadedLayer.Biases.cols,
			)
		}
	}

	// --- APPLICATION STEP ---
	// If we passed all checks, it is safe to overwrite the weights
	for i := range nw.Layers {
		loadedLayer := loadedData.LayerDatas[i]
		currentLayer := nw.Layers[i]

		copy(currentLayer.Weights.data, loadedLayer.Weights.data)
		copy(currentLayer.Biases.data, loadedLayer.Biases.data)
	}

	// Load Learning Rate
	nw.LearningRate = loadedData.LearningRate

	return nil
}

func (nw *NeuralNetwork) ComputeLossAndAccuracy(Y []float64) (float64, float64) {
	output := nw.Layers[len(nw.Layers)-1].A
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

// Predict takes a flattened image (1D slice), passes it through the network,
func (nw *NeuralNetwork) Predict(inputData []float64) (int, float64) {
	// 1. Validation
	inputSize := nw.Layers[0].Weights.rows
	if len(inputData) != inputSize {
		panic(fmt.Sprintf("Input size mismatch. Expected %d, got %d", inputSize, len(inputData)))
	}

	// Check if buffers exist and match the batch size (1)
	if nw.Layers[0].Z == nil || nw.Layers[0].Z.rows != 1 {
		nw.InitializeBuffers(1)
	}

	// Create Matrix View (1 Row, N Cols)
	// We treat the single image as a Batch of size 1.
	inputMat := NewMatrixFromSlice(1, inputSize, inputData)

	// 3. Run Forward Pass
	nw.Forward(inputMat)

	// 4. Read Output
	lastLayer := nw.Layers[len(nw.Layers)-1]
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

// ------- MATRIX METHODS ------ //
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

// ------ ADAM OPTIMIZER METHODS ------ //
// Update applies the Adam update rule to the network's weights and biases
func (opt *AdamOptimizer) Update(nw *NeuralNetwork, grads []GradientSet) {
	opt.timeStep++

	// Pre-calculate bias correction factors for efficiency
	// correction = 1 / (1 - beta^t)
	beta1Corr := 1.0 / (1.0 - math.Pow(opt.cfg.Beta1, float64(opt.timeStep)))
	beta2Corr := 1.0 / (1.0 - math.Pow(opt.cfg.Beta2, float64(opt.timeStep)))

	lr := opt.cfg.LearningRate

	for i, layer := range nw.Layers {
		state := opt.layerStates[i]

		// Update Weights
		applyAdam(
			layer.Weights.data,
			grads[i].dW.data,
			state.mW.data,
			state.vW.data,
			opt.cfg,
			lr,
			beta1Corr,
			beta2Corr,
		)

		// Update Biases
		applyAdam(
			layer.Biases.data,
			grads[i].db.data,
			state.mB.data,
			state.vB.data,
			opt.cfg,
			lr,
			beta1Corr,
			beta2Corr,
		)
	}
}

// applyAdam performs the element-wise math over the flat slice.
func applyAdam(
	params []float64,
	grads []float64,
	m []float64,
	v []float64,
	cfg AdamConfig,
	lr, beta1Corr, beta2Corr float64,
) {
	for i := range params {
		g := grads[i]

		// 1. Update biased first moment estimate (Momentum)
		// m_t = beta1 * m_{t-1} + (1 - beta1) * g
		m[i] = cfg.Beta1*m[i] + (1.0-cfg.Beta1)*g

		// 2. Update biased second raw moment estimate (Velocity)
		// v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
		v[i] = cfg.Beta2*v[i] + (1.0-cfg.Beta2)*(g*g)

		// 3. Compute bias-corrected moment estimates
		mHat := m[i] * beta1Corr
		vHat := v[i] * beta2Corr

		// 4. Update parameters
		// theta = theta - lr * mHat / (sqrt(vHat) + epsilon)
		params[i] -= lr * mHat / (math.Sqrt(vHat) + cfg.Epsilon)
	}
}

// ------ MOMENTUM OPTIMIZER METHODS ------ //
func (opt *MomentumOptimizer) Update(nw *NeuralNetwork, grads []GradientSet) {
	for i, layer := range nw.Layers {
		vW := opt.velocities[i][0].dW.data
		vB := opt.velocities[i][0].db.data
		dW := grads[i].dW.data
		db := grads[i].db.data

		// Update Velocity: v = mu*v - lr*grad
		floats.Scale(opt.Mu, vW)
		floats.AddScaled(vW, -opt.LearningRate, dW)

		floats.Scale(opt.Mu, vB)
		floats.AddScaled(vB, -opt.LearningRate, db)

		// Update Weights: W = W + v
		floats.Add(layer.Weights.data, vW)
		floats.Add(layer.Biases.data, vB)
	}
}

// ------ SGD OPTIMIZER METHODS ------ //
func (opt *SGDOptimizer) Update(nw *NeuralNetwork, grads []GradientSet) {
	for i, layer := range nw.Layers {
		// Simple update: W = W - (lr * gradient)
		floats.AddScaled(layer.Weights.data, -opt.LearningRate, grads[i].dW.data)
		floats.AddScaled(layer.Biases.data, -opt.LearningRate, grads[i].db.data)
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

// ------ DATA HANDLING HELPERS ------
func NewIndexList(size int) []int {
	indices := make([]int, size)
	for i := range indices {
		indices[i] = i
	}
	return indices
}

func ShuffleIndices(indices []int) {
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
		// We calculate where this specific input lives in the massive global array
		srcStart := realDataIdx * rowSize
		srcEnd := srcStart + rowSize

		// We calculate where it should go in the worker's small local buffer
		dstStart := localRowIdx * rowSize
		dstEnd := dstStart + rowSize

		copy(destX.data[dstStart:dstEnd], globalX[srcStart:srcEnd])
	}
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

// ------ IMAGE PROCESSING HELPERS ------
func convertJpg1D(path string, targetW, targetH int) ([]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	src, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}

	// Resize to 28x28 (or whatever the network expects)
	dst := image.NewRGBA(image.Rect(0, 0, targetW, targetH))
	draw.CatmullRom.Scale(dst, dst.Rect, src, src.Bounds(), draw.Over, nil)

	out := make([]float64, 0, targetW*targetH)
	bounds := dst.Bounds()

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := dst.At(x, y).RGBA()
			// Standard Grayscale formula
			gray := 0.299*float64(r>>8) + 0.587*float64(g>>8) + 0.114*float64(b>>8)
			out = append(out, gray) // Returns 0-255 range
		}
	}
	return out, nil
}
