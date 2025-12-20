package ml

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"
)

const (
	// Default tokens
	PAD = "<pad>"
	UNK = "<unk>"

	// Sampling types
	SamplingGreedy  = "greedy"
	SamplingUniform = "uniform"
	SamplingTopK    = "topk"
)

const (
	// ModeAlwaysSample (Case 1): Use the configured SamplingType for ALL tokens.
	ModeAlwaysSample DecodingMode = "always_sample"
	// ModeSampleFirstThenGreedy (Case 2): Use the configured SamplingType for the 1st token, then Greedy for the rest.
	ModeSampleFirstThenGreedy DecodingMode = "sample_first_then_greedy"
	// ModeIntervalSampling (Case 3): Use the configured SamplingType every 'Interval' tokens, otherwise Greedy.
	ModeIntervalSampling DecodingMode = "interval_sampling"
)

var endTokensInf = map[string]bool{
	".":   true,
	"?":   true,
	"!":   true,
	"eos": false,
}

// DecodingMode defines how frequently the configured SamplingType is used.
type DecodingMode string

// DecodingConfig holds the parameters for the inference decoding strategy.
type DecodingConfig struct {
	SamplingType string  // e.g., "greedy", "uniform", "topk" (Used for non-Greedy steps)
	Temperature  float64 // T > 0.
	TopK         int     // The K value for Top-K sampling.

	// New Fields for Mode Control
	Mode       DecodingMode // Defines when to use the SamplingType
	Interval   int          // Used only if Mode is ModeIntervalSampling (e.g., 10)
	Sequential bool         // Whether to use sequential inference (for LSTM)
}

// multinomialSample performs standard sampling from a probability distribution.
// Each class has a chance of being selected proportional to its probability.
func multinomialSample(probs []float64) int {
	r := rand.Float64()
	cumulativeProb := 0.0
	for i, p := range probs {
		cumulativeProb += p
		if r < cumulativeProb {
			return i
		}
	}
	// Fallback in case of floating point inaccuracies, return the last index.
	return len(probs) - 1
}

// greedySample finds the index of the maximum probability.
func greedySample(probs []float64) int {
	maxProb := -1.0
	maxIdx := 0
	for i, p := range probs {
		if p > maxProb {
			maxProb = p
			maxIdx = i
		}
	}
	return maxIdx
}

// topKSample zeros out probabilities outside the top K and then samples multinomial.
func topKSample(probs []float64, K int) int {
	numClasses := len(probs)
	if K <= 0 || K >= numClasses {
		// If K is invalid, fall back to standard multinomial sampling
		return multinomialSample(probs)
	}

	// 1. Find the K largest probabilities and their indices
	type probIndex struct {
		prob float64
		idx  int
	}

	// Create a list of all probabilities with their indices
	indexedProbs := make([]probIndex, numClasses)
	for i, p := range probs {
		indexedProbs[i] = probIndex{prob: p, idx: i}
	}

	// Sort in descending order by probability
	sort.Slice(indexedProbs, func(i, j int) bool {
		return indexedProbs[i].prob > indexedProbs[j].prob
	})

	// 2. Create the masked probabilities and calculate the new sum
	var topKProbs []float64
	var topKIndices []int
	newSum := 0.0

	// Only include the top K
	for i := range K {
		p := indexedProbs[i].prob
		topKProbs = append(topKProbs, p)
		topKIndices = append(topKIndices, indexedProbs[i].idx)
		newSum += p
	}

	// 3. Re-normalize the top K probabilities
	// If newSum is 0 (shouldn't happen with logit subtraction), fall back to uniform.
	if newSum == 0.0 {
		return multinomialSample(probs)
	}

	normalizedProbs := make([]float64, K)
	for i, p := range topKProbs {
		normalizedProbs[i] = p / newSum
	}

	// 4. Sample from the renormalized distribution
	sampledIndexInTopK := multinomialSample(normalizedProbs)

	// 5. Return the original class index
	return topKIndices[sampledIndexInTopK]
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

func InferenceImg(nw *NeuralNetwork, imagePath string, convertJpg1D func(string, int, int) ([]float64, error)) {
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
