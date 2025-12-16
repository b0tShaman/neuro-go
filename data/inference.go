package data

import (
	"math/rand"
	"sort"
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
