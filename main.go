package main

import (
	"fmt"
	"os"
	"runtime"

	"github.com/b0tShaman/neuro-go/data"
	. "github.com/b0tShaman/neuro-go/ml"
)

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
	X_global := NewMatrixFromSlice(len(X_raw), len(X_raw[0]), Flatten(X_raw))

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
		Epochs:       500,
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
