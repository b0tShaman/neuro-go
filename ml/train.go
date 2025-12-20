package ml

import (
	"fmt"
	"math/rand/v2"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"gonum.org/v1/gonum/floats"
)

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
