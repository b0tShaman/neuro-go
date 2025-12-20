package ml

import (
	"math"

	"gonum.org/v1/gonum/floats"
)

const (
	OptSGD      OptimizerType = "sgd"
	OptMomentum OptimizerType = "momentum"
	OptAdam     OptimizerType = "adam"
)

// Default settings generally recommended for Adam
var DefaultAdamConfig = AdamConfig{
	Beta1:        0.9,
	Beta2:        0.999,
	Epsilon:      1e-8,
	LearningRate: 0.001,
}

type OptimizerType string
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

type Optimizer interface {
	Update(nw *NeuralNetwork, grads []GradientSet)
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
