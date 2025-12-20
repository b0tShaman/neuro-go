package ml

import (
	"math"
)

const (
	ActLinear ActivationType = iota
	ActRelu
	ActSigmoid
	ActSoftmax
	ActEmbedding
	ActAttention
)

var activationMap = map[string]ActivationType{
	"linear":  ActLinear,
	"sigmoid": ActSigmoid,
	"relu":    ActRelu,
	"softmax": ActSoftmax,
}

// -------- TYPE DEFINITIONS -------- //
type ActivationType int
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

// AttentionWorkspace holds all pre-allocated matrices for a single sample's forward and backward pass.
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

// GradientSet holds the calculated gradients for one layer
type GradientSet struct {
	dW *Matrix
	db *Matrix
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

func Flatten(input [][]float64) []float64 {
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
