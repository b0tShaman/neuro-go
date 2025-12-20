package ml

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand/v2"
	"os"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type NeuralNetwork struct {
	Layers       []*Layer
	LearningRate float64
	InputT       *Matrix
	InputBuf     *Matrix
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
			// Input: [Batch, ContextLen * EmbedDim]
			embedDim := cfg.Neurons // This was set via SelfAttention(64)

			// Validation
			if prevOutputSize%embedDim != 0 {
				panic(fmt.Sprintf("Layer %d Input %d not divisible by EmbedDim %d", i, prevOutputSize, embedDim))
			}

			// Weights (W_O) are Square [EmbedDim, EmbedDim]
			weightsRows = embedDim
			weightsCols = embedDim
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

			// Calculate ContextLen based on Neurons (Total Output) / EmbedDim
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
