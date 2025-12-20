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
	ActAttention
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
	// Attention fields
	HeadCount int // Number of heads (Keeping it 1 for this implementation)
}

type LayerState struct {
	// Standard (Dense/Embedding/Projection)
	mW, vW *Matrix
	mB, vB *Matrix

	// Attention Specific
	mWQ, vWQ *Matrix
	mWK, vWK *Matrix
	mWV, vWV *Matrix
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

	// Attention Specific Weights
	WQ, WK, WV *Matrix

	// Attention Gradients
	dWQ, dWK, dWV *Matrix

	// Attention Cache (for Backprop)
	// We need to store Q, K, V per batch to calculate gradients
	// Storing as slice of Matrices because they differ per sample
	cacheQ, cacheK, cacheV []*Matrix

	PosEnc *Matrix // Stores the pre-computed positional encoding

	// Pre-allocated workspaces (Size = BatchSize)
	Workspaces []*AttentionWorkspace
}

// AttentionWorkspace holds all pre-allocated matrices for a single sample's
// Forward and Backward pass.
// AttentionWorkspace holds pre-allocated buffers for a single sample
type AttentionWorkspace struct {
	// Forward Buffers
	X, Q, K, V *Matrix
	Scores     *Matrix
	AttnOut    *Matrix

	// Backward Buffers
	dScores    *Matrix
	dAttnOut   *Matrix
	dQ, dK, dV *Matrix

	// Temporary Buffers (Reusable scratchpads for accumulation)
	tmpGrad *Matrix // Used for dWQ, dWK, dWV accumulation
	tmpX    *Matrix // Used for dX accumulation

	// NEW: Wrappers to avoid NewMatrixFromSlice inside loops
	dZ_proj    *Matrix // For reading incoming gradients in Backprop
	ProjectOut *Matrix // For writing outgoing results in Forward
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
	Mu           float64 // Momentum Factor (usually 0.9)

	// Changed from [][]GradientSet to support Attention (Q, K, V)
	layerStates []*LayerState
}

type TrainingConfig struct {
	Epochs       int
	BatchSize    int
	LearningRate float64
	ModelPath    string
	NumWorkers   int
	VerboseEvery int // How often to log progress (in epochs)

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
	prevOutputSize := configs[0].Neurons // Starts as ContextLen

	for i := 1; i < len(configs); i++ {
		cfg := configs[i]

		var weightsRows, weightsCols int
		var nextPrevOutputSize int // What we tell the NEXT layer our output is

		switch cfg.Activation {
		case ActEmbedding:
			if cfg.VocabSize == 0 {
				panic("VocabSize missing for Embedding")
			}

			weightsRows = cfg.VocabSize
			weightsCols = cfg.EmbedDim

			// Embedding expands Input(ContextLen) -> Output(ContextLen * EmbedDim)
			if cfg.Neurons == 0 {
				cfg.Neurons = prevOutputSize * cfg.EmbedDim
			}
			nextPrevOutputSize = cfg.Neurons

		case ActAttention:
			// [FIX] Attention Layer Dimensions
			// Input: [Batch, ContextLen * EmbedDim]
			// We need to extract EmbedDim from config
			embedDim := cfg.Neurons // This was set via SelfAttention(64)

			// Validation
			if prevOutputSize%embedDim != 0 {
				panic(fmt.Sprintf("Layer %d Input %d not divisible by EmbedDim %d", i, prevOutputSize, embedDim))
			}

			// Weights (W_O) are Square [EmbedDim, EmbedDim]
			weightsRows = embedDim
			weightsCols = embedDim

			// [FIX] Output size preserves Input size (ContextLen * EmbedDim)
			// We do NOT change prevOutputSize for the next layer
			nextPrevOutputSize = prevOutputSize

			// Force the layer's stored "Neurons" to match actual output for buffer init
			cfg.Neurons = prevOutputSize

		default:
			// Standard Dense Layer
			weightsRows = prevOutputSize
			weightsCols = cfg.Neurons
			nextPrevOutputSize = cfg.Neurons
		}

		layer := &Layer{
			Weights: NewMatrix(weightsRows, weightsCols),
			Biases:  NewMatrix(1, cfg.Neurons), // Biases match flattened output
			ActType: cfg.Activation,
		}

		// Initialize Specific Weights
		switch cfg.Activation {
		case ActAttention:
			embedDim := weightsRows
			layer.WQ = NewMatrix(embedDim, embedDim)
			layer.WQ.Randomize()
			layer.WK = NewMatrix(embedDim, embedDim)
			layer.WK.Randomize()
			layer.WV = NewMatrix(embedDim, embedDim)
			layer.WV.Randomize()

			layer.dWQ = NewMatrix(embedDim, embedDim)
			layer.dWK = NewMatrix(embedDim, embedDim)
			layer.dWV = NewMatrix(embedDim, embedDim)

			// Projection Weight (Weights) Init
			layer.Weights.Randomize()

		case ActEmbedding:
			layer.ActType = cfg.Activation

			// [ADD THIS BLOCK]
			// Calculate ContextLen based on Neurons (Total Output) / EmbedDim
			// Note: ensure cfg.Neurons is set correctly before this (it is set in your switch)
			contextLen := cfg.Neurons / cfg.EmbedDim

			peData := MakePositionalEncoding(contextLen, cfg.EmbedDim)
			layer.PosEnc = NewMatrixFromSlice(1, cfg.Neurons, peData)
			// Small init for embeddings
			for k := range layer.Weights.data {
				layer.Weights.data[k] = (rand.Float64()*2 - 1) * 0.01
			}
		default:
			layer.Weights.Randomize()
		}

		nn.Layers = append(nn.Layers, layer)
		prevOutputSize = nextPrevOutputSize
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
		layerStates:  make([]*LayerState, len(nw.Layers)),
	}

	// Pre-allocate memory for velocities (held in LayerState matrices)
	for i, layer := range nw.Layers {
		state := &LayerState{}

		// 1. Standard Velocities (Weights & Biases)
		// Used by Dense, Embedding, and Attention (for Output Projection)
		state.mW = NewMatrix(layer.Weights.rows, layer.Weights.cols)
		state.mB = NewMatrix(layer.Biases.rows, layer.Biases.cols)

		// 2. Attention Velocities (Q, K, V)
		if layer.ActType == ActAttention {
			r, c := layer.WQ.rows, layer.WQ.cols
			state.mWQ = NewMatrix(r, c)
			state.mWK = NewMatrix(r, c)
			state.mWV = NewMatrix(r, c)
		}

		opt.layerStates[i] = state
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

// Helper Constructor
func SelfAttention(embedDim int, opts ...LayerOption) LayerConfig {
	cfg := LayerConfig{
		Activation: ActAttention,
		Neurons:    embedDim, // Output dim usually equals input dim in Transformers
		IsInput:    false,
		HeadCount:  1,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
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
	// data.MinMaxNormalize(X_raw)

	// Text Data
	embedDim := 4
	contextLen := 5
	freqThreshold := 1 // Replace words with freq < threshold with <unk>

	var X_raw [][]float64
	var Y any
	var err error

	// outputDim := contextLen * embedDim // Output dimension for Embedding layer for non-sequential model(n-gram)

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
		Input(contextLen),
		Embedding(embedDim, VocabSize(vocabSize)), // Output: Batch x (ContextLen * 64)
		SelfAttention(embedDim),                   // Input: (ContextLen*64) -> Output: (ContextLen*64)
		Dense(64),
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
		Epochs:       1000,
		BatchSize:    G * 16,
		LearningRate: 0.001, // Adam default is 0.001, SGD can be higher
		ModelPath:    modelFile,
		NumWorkers:   G,
		Optimizer:    OptAdam,
		MomentumMu:   0.9, // Only used if OptMomentum is selected
		VerboseEvery: 100,
	}

	fmt.Printf("Running on %d cores (Mini-Batch = %d)\n\n", runtime.GOMAXPROCS(0), config.BatchSize)

	// Text Training and Inference
	Train(nw, X_global, Y_raw, config)
	InferenceTxt(nw, wordToID, idToWord)

	// Img Training and Inference
	// Train(nw, X_global, Y_raw, config)
	// InferenceImg(nw, "assets/5.jpg")
}

func Train(nw *NeuralNetwork, X_ids *Matrix, Y_ids []float64, cfg TrainingConfig) {
	fmt.Printf("TrainingConfig: %+v\n", cfg)
	validateConfig(cfg)

	// 1. Setup & Allocation
	optimizer := NewOptimizer(nw, cfg)
	localBatchSize := cfg.BatchSize / cfg.NumWorkers
	inputCtxLen := X_ids.cols
	numSamples := X_ids.rows

	// Delegate allocation logic to helpers
	workers, workerGrads := initializeWorkers(nw, cfg.NumWorkers, localBatchSize)
	finalGrads := initializeMasterGradients(nw)
	workerLabels, workerLosses, workerAccs := initializeAuxBuffers(cfg.NumWorkers, localBatchSize)
	globalIndices := NewIndexList(numSamples)

	// Delegate signal handling
	setupSignalHandler(nw, cfg.ModelPath)

	// 2. Training Loop
	start := time.Now()
	fmt.Println("Starting Training...")

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		ShuffleIndices(globalIndices)

		var totalLoss, totalAcc float64
		batchesProcessed := 0

		for batchStart := 0; batchStart+cfg.BatchSize <= numSamples; batchStart += cfg.BatchSize {
			var wg sync.WaitGroup
			wg.Add(cfg.NumWorkers)

			// --- A. Data Parallelism: Dispatch Workers ---
			for i := 0; i < cfg.NumWorkers; i++ {
				go func(id int) {
					defer wg.Done()
					wStart := batchStart + (id * localBatchSize)
					wEnd := wStart + localBatchSize
					myIndices := globalIndices[wStart:wEnd]

					// Copy data to worker buffers
					Gather(myIndices, X_ids.data, Y_ids, inputCtxLen, workers[id].InputBuf, workerLabels[id])

					// Forward & Backward pass
					workers[id].Forward(workers[id].InputBuf)
					loss, acc := workers[id].ComputeGradients(workers[id].InputBuf, workerLabels[id], workerGrads[id])

					workerLosses[id] = loss
					workerAccs[id] = acc
				}(i)
			}
			wg.Wait()

			// --- B. Aggregation Logic ---
			scale := 1.0 / float64(cfg.NumWorkers)

			// 1. Aggregate Standard Weights & Biases
			for l := range finalGrads {
				finalDW := finalGrads[l].dW
				finalDB := finalGrads[l].db

				// Initialize with Worker 0
				copy(finalDW.data, workerGrads[0][l].dW.data)
				copy(finalDB.data, workerGrads[0][l].db.data)

				// Sum remaining workers
				for w := 1; w < cfg.NumWorkers; w++ {
					floats.Add(finalDW.data, workerGrads[w][l].dW.data)
					floats.Add(finalDB.data, workerGrads[w][l].db.data)
				}

				// Apply Scale
				floats.Scale(scale, finalDW.data)
				floats.Scale(scale, finalDB.data)

				// 2. Aggregate Attention Specifics (WQ, WK, WV) if applicable
				if nw.Layers[l].ActType == ActAttention {
					aggregateAttentionGradients(nw.Layers[l], workers, l, cfg.NumWorkers, scale)
				}
			}

			// --- C. Optimization & Tracking ---
			optimizer.Update(nw, finalGrads)

			for i := range cfg.NumWorkers {
				totalLoss += workerLosses[i] * scale // equivalent to / NumWorkers
				totalAcc += workerAccs[i] * scale
			}
			batchesProcessed++
		}

		// Logging
		avgLoss := totalLoss / float64(batchesProcessed)
		avgAcc := (totalAcc / float64(batchesProcessed)) * 100
		if epoch%cfg.VerboseEvery == 0 || epoch == 1 {
			fmt.Printf("Epoch %d | Loss: %.4f | Acc: %.2f%% | Time: %v\n", epoch, avgLoss, avgAcc, time.Since(start))
		}
	}

	nw.SaveToFile(cfg.ModelPath)
	fmt.Printf("Training Complete. Total Time: %v\n\n", time.Since(start))
}

func InferenceTxt(nw *NeuralNetwork, wordToID map[string]int, idToWord []string) {
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
	const EOS = "." // Stop generation at this token

	var endTokensInf = map[string]bool{
		".":   false,
		"?":   false,
		"!":   false,
		"eos": false,
	}

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
			if word == "<crlf>" {
				fmt.Print("\n")
			} else {
				fmt.Print(word + " ")
			}

			time.Sleep(100 * time.Millisecond)

			// F. Stop Conditions
			if word == EOS {
				break
			}

			// G. Shift Window (Sliding Context)
			// [A, B, C] -> [B, C, NewID]
			copy(window[0:], window[1:])
			window[contextLen-1] = float64(bestID)

			// 2. FULL STOP logic
			if endTokensInf[word] {
				fmt.Print("\n\nEnter text: ")

				newLine, _ := reader.ReadString('\n')
				newLine = strings.TrimSpace(strings.ToLower(newLine))

				if newLine == "" {
					break
				}

				newWords := strings.Fields(newLine)

				// Append new text into window sliding
				for _, w := range newWords {
					id, ok := wordToID[w]
					if !ok {
						id = wordToID[UNK]
					}
					window = append(window[1:], float64(id))
				}

				// Continue generating based on updated window
				continue
			}
		}
		fmt.Println() // Newline after response
	}
}

func InferenceImg(nw *NeuralNetwork, imagePath string) {
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

func validateConfig(cfg TrainingConfig) {
	if cfg.BatchSize%cfg.NumWorkers != 0 {
		panic("BatchSize must be divisible by NumWorkers")
	}
}

// initializeWorkers creates clones of the network and allocates gradient memory for each worker
func initializeWorkers(nw *NeuralNetwork, numWorkers, localBatchSize int) ([]*NeuralNetwork, [][]GradientSet) {
	fmt.Printf("Initializing %d workers (Worker Batch: %d)\n", numWorkers, localBatchSize)

	workers := make([]*NeuralNetwork, numWorkers)
	workerGrads := make([][]GradientSet, numWorkers)

	for i := 0; i < numWorkers; i++ {
		workers[i] = nw.CloneStructure()
		workers[i].InitializeBuffers(localBatchSize)

		workerGrads[i] = make([]GradientSet, len(nw.Layers))
		for l := 0; l < len(nw.Layers); l++ {
			// Standard Gradients
			wRows, wCols := nw.Layers[l].Weights.rows, nw.Layers[l].Weights.cols
			bRows, bCols := nw.Layers[l].Biases.rows, nw.Layers[l].Biases.cols

			workerGrads[i][l].dW = NewMatrix(wRows, wCols)
			workerGrads[i][l].db = NewMatrix(bRows, bCols)

			// Special Attention Gradients (stored in Layer struct, not GradientSet)
			if nw.Layers[l].ActType == ActAttention {
				attnDim := wRows
				workers[i].Layers[l].dWQ = NewMatrix(attnDim, attnDim)
				workers[i].Layers[l].dWK = NewMatrix(attnDim, attnDim)
				workers[i].Layers[l].dWV = NewMatrix(attnDim, attnDim)
			}
		}
	}
	return workers, workerGrads
}

// initializeMasterGradients allocates the buffer for aggregated gradients
func initializeMasterGradients(nw *NeuralNetwork) []GradientSet {
	finalGrads := make([]GradientSet, len(nw.Layers))
	for l := 0; l < len(nw.Layers); l++ {
		wRows, wCols := nw.Layers[l].Weights.rows, nw.Layers[l].Weights.cols
		bRows, bCols := nw.Layers[l].Biases.rows, nw.Layers[l].Biases.cols
		finalGrads[l].dW = NewMatrix(wRows, wCols)
		finalGrads[l].db = NewMatrix(bRows, bCols)
	}
	return finalGrads
}

// initializeAuxBuffers creates buffers for labels, losses, and accuracy
func initializeAuxBuffers(numWorkers, localBatchSize int) ([][]float64, []float64, []float64) {
	workerLabels := make([][]float64, numWorkers)
	for i := 0; i < numWorkers; i++ {
		workerLabels[i] = make([]float64, localBatchSize)
	}
	workerLosses := make([]float64, numWorkers)
	workerAccs := make([]float64, numWorkers)
	return workerLabels, workerLosses, workerAccs
}

// setupSignalHandler captures SIGINT/SIGTERM to save the model safely
func setupSignalHandler(nw *NeuralNetwork, modelPath string) {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigChan
		fmt.Println("\nInterrupt! Saving model...")
		nw.SaveToFile(modelPath)
		os.Exit(0)
	}()
}

// aggregateAttentionGradients handles the specific summation logic for Transformer layers
func aggregateAttentionGradients(masterLayer *Layer, workers []*NeuralNetwork, layerIdx int, numWorkers int, scale float64) {
	// 1. Reset Master Gradients
	masterLayer.dWQ.Reset()
	masterLayer.dWK.Reset()
	masterLayer.dWV.Reset()

	// 2. Sum up gradients from all workers
	for w := 0; w < numWorkers; w++ {
		workerLayer := workers[w].Layers[layerIdx]
		floats.Add(masterLayer.dWQ.data, workerLayer.dWQ.data)
		floats.Add(masterLayer.dWK.data, workerLayer.dWK.data)
		floats.Add(masterLayer.dWV.data, workerLayer.dWV.data)
	}

	// 3. Apply Scaling
	scaleFunc := func(v float64) float64 { return v * scale }
	masterLayer.dWQ.ApplyFunc(scaleFunc)
	masterLayer.dWK.ApplyFunc(scaleFunc)
	masterLayer.dWV.ApplyFunc(scaleFunc)
}

// -------- NEURAL NETWORK METHODS -------- //
func (nw *NeuralNetwork) InitializeBuffers(batchSize int) {
	// 1. Determine Input Dimensions for Layer 0
	inputDim := nw.Layers[0].Weights.rows
	if nw.Layers[0].ActType == ActEmbedding {
		// For Embedding, inputDim is calculated via biases or config
		inputDim = nw.Layers[0].Biases.cols / nw.Layers[0].Weights.cols
	}

	// 2. Global Input Buffers
	data := make([]float64, batchSize*inputDim)
	nw.InputBuf = &Matrix{
		rows:  batchSize,
		cols:  inputDim,
		data:  data,
		dense: mat.NewDense(batchSize, inputDim, data),
	}
	nw.InputT = NewMatrix(inputDim, batchSize)

	// 3. Loop Layers
	for _, layer := range nw.Layers {
		var outputDim int

		// A. Determine Output Size
		if layer.ActType == ActEmbedding || layer.ActType == ActAttention {
			outputDim = layer.Biases.cols
		} else {
			outputDim = layer.Weights.cols
		}

		// B. Standard Layer Buffers
		layer.Z = NewMatrix(batchSize, outputDim)
		layer.A = NewMatrix(batchSize, outputDim)
		layer.dZ = NewMatrix(batchSize, outputDim)

		// C. Attention Specific Workspaces (ZERO ALLOC SETUP)
		if layer.ActType == ActAttention {
			embedDim := layer.Weights.rows
			totalCols := layer.Biases.cols // This is (ContextLen * EmbedDim)
			contextLen := totalCols / embedDim

			// Create a workspace for every sample in the batch
			layer.Workspaces = make([]*AttentionWorkspace, batchSize)

			for b := 0; b < batchSize; b++ {
				ws := &AttentionWorkspace{}

				// 1. Forward Matrices
				ws.X = NewMatrix(contextLen, embedDim)
				ws.Q = NewMatrix(contextLen, embedDim)
				ws.K = NewMatrix(contextLen, embedDim)
				ws.V = NewMatrix(contextLen, embedDim)
				ws.Scores = NewMatrix(contextLen, contextLen)
				ws.AttnOut = NewMatrix(contextLen, embedDim)

				// 2. Backward Matrices
				ws.dScores = NewMatrix(contextLen, contextLen)
				ws.dAttnOut = NewMatrix(contextLen, embedDim)
				ws.dQ = NewMatrix(contextLen, embedDim)
				ws.dK = NewMatrix(contextLen, embedDim)
				ws.dV = NewMatrix(contextLen, embedDim)

				// 3. Scratchpads
				// Size matches the largest matrix we need to accumulate (EmbedDim x EmbedDim)
				ws.tmpGrad = NewMatrix(embedDim, embedDim)
				ws.tmpX = NewMatrix(contextLen, embedDim)

				// They are size [ContextLen, EmbedDim]
				ws.dZ_proj = NewMatrix(contextLen, embedDim)
				ws.ProjectOut = NewMatrix(contextLen, embedDim)

				layer.Workspaces[b] = ws
			}
		}
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
			WQ:      l.WQ,
			WK:      l.WK,
			WV:      l.WV,
			PosEnc:  l.PosEnc,
		}
	}
	return newNN
}

func (nw *NeuralNetwork) Forward(input *Matrix) {
	activation := input
	for _, layer := range nw.Layers {
		switch layer.ActType {
		case ActAttention:
			// --- ATTENTION FORWARD (Sequential & Zero Alloc) ---
			batchSize := activation.rows
			totalInputCols := activation.cols
			embedDim := layer.Weights.rows
			contextLen := totalInputCols / embedDim
			scale := 1.0 / math.Sqrt(float64(embedDim))

			// Loop sequentially (No go routines) to avoid oversubscription
			for b := 0; b < batchSize; b++ {

				// 1. Get Workspace
				ws := layer.Workspaces[b]

				// 2. Load Input (Copy slice to Workspace X)
				rowStart := b * totalInputCols
				copy(ws.X.data, activation.data[rowStart:rowStart+totalInputCols])

				// 3. Compute Q, K, V
				// MatMul stores result directly in ws.Q, ws.K, ws.V
				MatMul(ws.X.dense, layer.WQ.dense, ws.Q)
				MatMul(ws.X.dense, layer.WK.dense, ws.K)
				MatMul(ws.X.dense, layer.WV.dense, ws.V)

				// 4. Compute Scores = Q * K^T
				MatMul(ws.Q.dense, ws.K.dense.T(), ws.Scores)

				// 5. Causal Masking & Scaling (Direct Data Access)
				sData := ws.Scores.data
				for r := 0; r < contextLen; r++ {
					rowOffset := r * contextLen
					for c := 0; c < contextLen; c++ {
						if c > r {
							sData[rowOffset+c] = -1e9 // Mask Future
						} else {
							sData[rowOffset+c] *= scale
						}
					}
				}

				// 6. Softmax (In-place on ws.Scores)
				SoftmaxRow(ws.Scores)

				// 7. Attention Output = Scores * V
				MatMul(ws.Scores.dense, ws.V.dense, ws.AttnOut)

				// 8. Output Projection (AttnOut * W_O) -> Global A
				// We create a view of the destination row in layer.A
				// STEP A: Compute into pre-allocated Workspace buffer (Zero Alloc)
				MatMul(ws.AttnOut.dense, layer.Weights.dense, ws.ProjectOut)

				// STEP B: Copy raw floats to Global A (Zero Alloc, very fast)
				destStart := b * totalInputCols
				// We copy from ws.ProjectOut.data to layer.A.data
				copy(layer.A.data[destStart:destStart+totalInputCols], ws.ProjectOut.data)
			}

			// Pass forward
			activation = layer.A

		case ActEmbedding:
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
			// Add Positional Encoding to every sample in the batch
			peData := layer.PosEnc.data
			outData = layer.A.data

			// Since outData is [BatchSize, ContextLen * EmbedDim]
			// and peData is [1, ContextLen * EmbedDim]
			// We broadcast peData across every row of outData

			rowSize := layer.PosEnc.cols // ContextLen * EmbedDim

			for b := 0; b < batchSize; b++ {
				offset := b * rowSize
				for i := 0; i < rowSize; i++ {
					outData[offset+i] += peData[i]
				}
			}
			// Embedding has no bias or activation function, strictly lookup
			activation = layer.A

		default:
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

		if layer.ActType == ActAttention {
			// --- ATTENTION BACKWARD (Sequential & Zero Alloc) ---

			// 1. Reset Gradients for this batch
			layer.dWQ.Reset()
			layer.dWK.Reset()
			layer.dWV.Reset()
			grads[i].dW.Reset() // dWO (Projection Weight)

			embedDim := layer.Weights.rows
			totalCols := layer.dZ.cols
			contextLen := totalCols / embedDim
			scale := 1.0 / math.Sqrt(float64(embedDim))

			// Determine if we need to calc dZ_prev for previous layer
			calcPrev := i > 0
			var dX_prev_data []float64
			if calcPrev {
				nw.Layers[i-1].dZ.Reset()
				dX_prev_data = nw.Layers[i-1].dZ.data
			}

			// Loop Sequentially
			for b := 0; b < int(batchSize); b++ {
				ws := layer.Workspaces[b]

				// Incoming Gradient (dZ) from next layer
				// We wrap the slice from global dZ buffer
				dZStart := b * totalCols

				// BAD (Allocates):
				// dZ_proj := NewMatrixFromSlice(contextLen, embedDim, layer.dZ.data[dZStart : dZStart+totalCols])

				// GOOD (Zero Alloc):
				// Copy the relevant slice from global dZ into our workspace
				copy(ws.dZ_proj.data, layer.dZ.data[dZStart:dZStart+totalCols])

				// --- A. Backprop Output Projection (W_O) ---

				// 1. dAttnOut = dZ_proj * W_O^T
				MatMul(ws.dZ_proj.dense, layer.Weights.dense.T(), ws.dAttnOut)

				// 2. dWO += AttnOut^T * dZ_proj
				// We calculate into tmpGrad, then add to global accumulator
				MatMul(ws.AttnOut.dense.T(), ws.dZ_proj.dense, ws.tmpGrad)
				floats.Add(grads[i].dW.data, ws.tmpGrad.data)

				// --- B. Backprop V ---

				// 1. dV = Scores^T * dAttnOut
				MatMul(ws.Scores.dense.T(), ws.dAttnOut.dense, ws.dV)

				// 2. dWV += X^T * dV
				MatMul(ws.X.dense.T(), ws.dV.dense, ws.tmpGrad)
				floats.Add(layer.dWV.data, ws.tmpGrad.data)

				// --- C. Backprop Scores (Softmax Derivative) ---

				// 1. dScoresRaw = dAttnOut * V^T
				MatMul(ws.dAttnOut.dense, ws.V.dense.T(), ws.dScores)

				// 2. Apply Softmax Derivative In-Place
				// dS_i = S_i * (dScoresRaw_i - dot(S, dScoresRaw))
				scoresData := ws.Scores.data
				dScoresData := ws.dScores.data

				for r := 0; r < contextLen; r++ {
					start := r * contextLen
					end := start + contextLen

					// Calculate Dot Product for this row
					dot := 0.0
					for k := start; k < end; k++ {
						dot += scoresData[k] * dScoresData[k]
					}

					// Update dScores
					for k := start; k < end; k++ {
						val := scoresData[k] * (dScoresData[k] - dot)
						dScoresData[k] = val * scale // Apply Scale Factor Here
					}
				}

				// --- D. Backprop Q & K ---

				// 1. dQ = dScores * K
				MatMul(ws.dScores.dense, ws.K.dense, ws.dQ)

				// 2. dK = dScores^T * Q
				MatMul(ws.dScores.dense.T(), ws.Q.dense, ws.dK)

				// 3. dWQ += X^T * dQ
				MatMul(ws.X.dense.T(), ws.dQ.dense, ws.tmpGrad)
				floats.Add(layer.dWQ.data, ws.tmpGrad.data)

				// 4. dWK += X^T * dK
				MatMul(ws.X.dense.T(), ws.dK.dense, ws.tmpGrad)
				floats.Add(layer.dWK.data, ws.tmpGrad.data)

				// --- E. Backprop Input X (Accumulate into prev layer dZ) ---
				if calcPrev {
					// dX = dQ*WQ^T + dK*WK^T + dV*WV^T

					// Term 1 (Q) -> Accumulate into ws.tmpX
					MatMul(ws.dQ.dense, layer.WQ.dense.T(), ws.tmpX)

					// Term 2 (K) -> Store in tmpGrad for a moment (repurposing)
					// Note: tmpGrad is [EmbedDim, EmbedDim], but we need [ContextLen, EmbedDim]
					// We must assume ws.tmpGrad is NOT large enough for this if ContextLen > EmbedDim.
					// SAFE FIX: Use ws.Q and ws.K as temporary buffers now, since we are done with their forward values!

					// Term 2 -> ws.Q (reused as temp)
					MatMul(ws.dK.dense, layer.WK.dense.T(), ws.Q)
					floats.Add(ws.tmpX.data, ws.Q.data)

					// Term 3 (V) -> ws.K (reused as temp)
					MatMul(ws.dV.dense, layer.WV.dense.T(), ws.K)
					floats.Add(ws.tmpX.data, ws.K.data)

					// Write to Global dZ_prev
					destSlice := dX_prev_data[b*totalCols : (b+1)*totalCols]
					floats.Add(destSlice, ws.tmpX.data)
				}
			}

			// Average Gradients by batch size
			batchScale := 1.0 / float64(batchSize)
			layer.dWQ.ApplyFunc(func(v float64) float64 { return v * batchScale })
			layer.dWK.ApplyFunc(func(v float64) float64 { return v * batchScale })
			layer.dWV.ApplyFunc(func(v float64) float64 { return v * batchScale })
			// grads[i].dW (dWO) is scaled later in the common code

		} else if layer.ActType == ActEmbedding {
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

	// Updated Struct to hold Attention & Embedding extras
	type LayerData struct {
		Weights *Matrix
		Biases  *Matrix
		ActType ActivationType

		// Attention Specific
		WQ, WK, WV *Matrix

		// Embedding Specific (Positional Encoding)
		PosEnc *Matrix
	}

	type NetworkData struct {
		LayerDatas   []LayerData
		LearningRate float64
	}

	ld := make([]LayerData, len(nw.Layers))
	for i, l := range nw.Layers {
		data := LayerData{
			Weights: l.Weights,
			Biases:  l.Biases,
			ActType: l.ActType,
		}

		// Save Attention Weights only if they exist
		if l.ActType == ActAttention {
			data.WQ = l.WQ
			data.WK = l.WK
			data.WV = l.WV
		}

		// Save Positional Encoding if it exists (Embedding Layer)
		if l.ActType == ActEmbedding {
			data.PosEnc = l.PosEnc
		}

		ld[i] = data
	}

	fmt.Println("Saving model to", filename)
	return encoder.Encode(NetworkData{LayerDatas: ld, LearningRate: nw.LearningRate})
}

func (nw *NeuralNetwork) LoadFromFile(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	decoder := gob.NewDecoder(file)

	// Same struct definition as SaveToFile
	type LayerData struct {
		Weights    *Matrix
		Biases     *Matrix
		ActType    ActivationType
		WQ, WK, WV *Matrix
		PosEnc     *Matrix
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

	if len(nw.Layers) != len(loadedData.LayerDatas) {
		return fmt.Errorf("architecture mismatch: current network has %d layers, model file has %d",
			len(nw.Layers), len(loadedData.LayerDatas))
	}

	// Helper to check matrix dimensions
	checkDims := func(name string, layerIdx int, current, loaded *Matrix) error {
		if current == nil && loaded == nil {
			return nil
		}
		if current == nil || loaded == nil {
			return fmt.Errorf("layer %d %s mismatch: one is nil", layerIdx, name)
		}
		if current.rows != loaded.rows || current.cols != loaded.cols {
			return fmt.Errorf("layer %d %s shape mismatch: expected [%d, %d], got [%d, %d]",
				layerIdx, name,
				current.rows, current.cols,
				loaded.rows, loaded.cols,
			)
		}
		return nil
	}

	for i, currLayer := range nw.Layers {
		loadedLayer := loadedData.LayerDatas[i]

		// 1. Basic Checks
		if currLayer.ActType != loadedLayer.ActType {
			return fmt.Errorf("layer %d mismatch: expected activation %v, got %v",
				i, currLayer.ActType, loadedLayer.ActType)
		}
		if err := checkDims("Weights", i, currLayer.Weights, loadedLayer.Weights); err != nil {
			return err
		}
		if err := checkDims("Biases", i, currLayer.Biases, loadedLayer.Biases); err != nil {
			return err
		}

		// 2. Attention Checks
		if currLayer.ActType == ActAttention {
			if err := checkDims("WQ", i, currLayer.WQ, loadedLayer.WQ); err != nil {
				return err
			}
			if err := checkDims("WK", i, currLayer.WK, loadedLayer.WK); err != nil {
				return err
			}
			if err := checkDims("WV", i, currLayer.WV, loadedLayer.WV); err != nil {
				return err
			}
		}

		// 3. Embedding Checks
		if currLayer.ActType == ActEmbedding {
			if err := checkDims("PosEnc", i, currLayer.PosEnc, loadedLayer.PosEnc); err != nil {
				return err
			}
		}
	}

	// --- APPLICATION STEP ---
	// Safe to overwrite now
	for i := range nw.Layers {
		loadedLayer := loadedData.LayerDatas[i]
		currentLayer := nw.Layers[i]

		// Standard
		copy(currentLayer.Weights.data, loadedLayer.Weights.data)
		copy(currentLayer.Biases.data, loadedLayer.Biases.data)

		// Attention
		if currentLayer.ActType == ActAttention {
			// Because WQ, WK, WV were allocated in NewNetwork(), we just copy data
			copy(currentLayer.WQ.data, loadedLayer.WQ.data)
			copy(currentLayer.WK.data, loadedLayer.WK.data)
			copy(currentLayer.WV.data, loadedLayer.WV.data)
		}

		// Embedding
		if currentLayer.ActType == ActEmbedding {
			copy(currentLayer.PosEnc.data, loadedLayer.PosEnc.data)
		}
	}

	nw.LearningRate = loadedData.LearningRate
	fmt.Println("Weights loaded successfully.")
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

// ------ ADAM OPTIMIZER METHODS ------ //
// Update applies the Adam update rule to the network's weights and biases
func (opt *AdamOptimizer) Update(nw *NeuralNetwork, grads []GradientSet) {
	// 1. Increment Time Step
	opt.timeStep++
	t := float64(opt.timeStep)

	// 2. Pre-calculate Correction Factors
	// correction1 = 1 - beta1^t
	// correction2 = 1 - beta2^t
	correction1 := 1.0 - math.Pow(opt.cfg.Beta1, t)
	correction2 := 1.0 - math.Pow(opt.cfg.Beta2, t)

	// Ensure State Storage exists
	if len(opt.layerStates) != len(nw.Layers) {
		opt.layerStates = make([]LayerState, len(nw.Layers))
	}

	// --- Helper Closure for Adam Math ---
	// Applies update to a single parameter vector (weights, bias, or Q/K/V)
	apply := func(params, grads, m, v []float64) {
		beta1 := opt.cfg.Beta1
		beta2 := opt.cfg.Beta2
		eps := opt.cfg.Epsilon
		lr := opt.cfg.LearningRate

		for i := range params {
			g := grads[i]

			// Update Moving Averages
			// m_t = beta1 * m_{t-1} + (1 - beta1) * g
			m[i] = beta1*m[i] + (1.0-beta1)*g

			// v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
			v[i] = beta2*v[i] + (1.0-beta2)*(g*g)

			// Bias Correction
			mHat := m[i] / correction1
			vHat := v[i] / correction2

			// Update Parameters
			// theta = theta - lr * mHat / (sqrt(vHat) + eps)
			params[i] -= lr * mHat / (math.Sqrt(vHat) + eps)
		}
	}

	// 3. Loop Layers
	for i, layer := range nw.Layers {
		state := &opt.layerStates[i]

		// --- A. Handle Attention Layers (Q, K, V) ---
		if layer.ActType == ActAttention {
			// Initialize Attention States if needed
			if state.mWQ == nil {
				rows, cols := layer.WQ.rows, layer.WQ.cols
				state.mWQ = NewMatrix(rows, cols)
				state.vWQ = NewMatrix(rows, cols)
				state.mWK = NewMatrix(rows, cols)
				state.vWK = NewMatrix(rows, cols)
				state.mWV = NewMatrix(rows, cols)
				state.vWV = NewMatrix(rows, cols)
			}

			// Apply Adam to Q, K, V
			// Note: We use layer.dWQ (Accumulated Master Gradients)
			apply(layer.WQ.data, layer.dWQ.data, state.mWQ.data, state.vWQ.data)
			apply(layer.WK.data, layer.dWK.data, state.mWK.data, state.vWK.data)
			apply(layer.WV.data, layer.dWV.data, state.mWV.data, state.vWV.data)
		}

		// --- B. Handle Standard Weights (Dense / Embedding / Projection) ---
		if state.mW == nil {
			state.mW = NewMatrix(layer.Weights.rows, layer.Weights.cols)
			state.vW = NewMatrix(layer.Weights.rows, layer.Weights.cols)
			state.mB = NewMatrix(layer.Biases.rows, layer.Biases.cols)
			state.vB = NewMatrix(layer.Biases.rows, layer.Biases.cols)
		}

		// Apply to Weights (All layers have Weights)
		apply(layer.Weights.data, grads[i].dW.data, state.mW.data, state.vW.data)

		// Apply to Biases (Embeddings usually don't have bias updates)
		if layer.ActType != ActEmbedding {
			apply(layer.Biases.data, grads[i].db.data, state.mB.data, state.vB.data)
		}
	}
}

// ------ MOMENTUM OPTIMIZER METHODS ------ //
func (opt *MomentumOptimizer) Update(nw *NeuralNetwork, grads []GradientSet) {
	// 1. Initialize State if needed
	if len(opt.layerStates) != len(nw.Layers) {
		opt.layerStates = make([]*LayerState, len(nw.Layers))
	}

	// Helper closure to apply Momentum:
	// v = mu * v - lr * grad
	// w = w + v
	applyMomentum := func(params, grads, velocity []float64) {
		for i := range params {
			// Update Velocity
			velocity[i] = (opt.Mu * velocity[i]) - (opt.LearningRate * grads[i])
			// Update Parameter
			params[i] += velocity[i]
		}
	}

	for i, layer := range nw.Layers {
		if opt.layerStates[i] == nil {
			opt.layerStates[i] = &LayerState{}
		}
		state := opt.layerStates[i]

		// --- A. Handle Attention Layers (Q, K, V) ---
		if layer.ActType == ActAttention {
			// Ensure Velocity Matrices exist
			if state.mWQ == nil {
				r, c := layer.WQ.rows, layer.WQ.cols
				// We use mW fields to store Velocity for consistency with LayerState struct
				state.mWQ = NewMatrix(r, c)
				state.mWK = NewMatrix(r, c)
				state.mWV = NewMatrix(r, c)
			}

			// Apply to Q, K, V using the MASTER accumulated gradients (layer.dWQ, etc.)
			applyMomentum(layer.WQ.data, layer.dWQ.data, state.mWQ.data)
			applyMomentum(layer.WK.data, layer.dWK.data, state.mWK.data)
			applyMomentum(layer.WV.data, layer.dWV.data, state.mWV.data)
		}

		// --- B. Handle Standard Weights (Dense / Embedding / Projection) ---
		if state.mW == nil {
			state.mW = NewMatrix(layer.Weights.rows, layer.Weights.cols)
			state.mB = NewMatrix(layer.Biases.rows, layer.Biases.cols)
		}

		// Update Weights (All layers)
		applyMomentum(layer.Weights.data, grads[i].dW.data, state.mW.data)

		// Update Biases (Skip for Embedding as they are usually unused/fixed)
		if layer.ActType != ActEmbedding {
			applyMomentum(layer.Biases.data, grads[i].db.data, state.mB.data)
		}
	}
}

// ------ SGD OPTIMIZER METHODS ------ //
func (opt *SGDOptimizer) Update(nw *NeuralNetwork, grads []GradientSet) {
	for i, layer := range nw.Layers {
		// 1. Update Attention Matrices (Q, K, V)
		if layer.ActType == ActAttention {
			// Now layer.dWQ contains the averaged gradients from all workers
			lr := opt.LearningRate // Use Optimizer's LR

			floats.AddScaled(layer.WQ.data, -lr, layer.dWQ.data)
			floats.AddScaled(layer.WK.data, -lr, layer.dWK.data)
			floats.AddScaled(layer.WV.data, -lr, layer.dWV.data)

			// Note: layer.Weights (Projector) is updated in step 2 below
		}
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

// MakePositionalEncoding creates a flattened vector of size [ContextLen * EmbedDim]
// containing the standard sinusoidal timing signals.
func MakePositionalEncoding(contextLen, embedDim int) []float64 {
	pe := make([]float64, contextLen*embedDim)

	for pos := 0; pos < contextLen; pos++ {
		for i := 0; i < embedDim; i++ {
			// Standard Formula:
			// PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
			// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

			exponent := float64(2*(i/2)) / float64(embedDim)
			divTerm := math.Pow(10000.0, exponent)
			val := float64(pos) / divTerm

			var encoding float64
			if i%2 == 0 {
				encoding = math.Sin(val)
			} else {
				encoding = math.Cos(val)
			}

			// Map 2D [pos, i] to 1D flat index
			idx := (pos * embedDim) + i
			pe[idx] = encoding
		}
	}
	return pe
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
