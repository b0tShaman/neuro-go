package main

import (
    "bytes"
    "encoding/gob"
    "fmt"
    "image/jpeg"
    "math"
    "math/rand"
    "os"
    "os/signal"
    "runtime"
    "sync"
    "syscall"
    "time"

    "github.com/b0tShaman/go-rout-net/data"
    // Assume mnist package is available
)

// --- Matrix Library ---

// Matrix represents a dense matrix with a flat data slice for performance.
type Matrix struct {
    rows, cols int
    data       []float64
}

// NewMatrix creates a matrix of zeros.
func NewMatrix(rows, cols int) *Matrix {
    return &Matrix{
        rows: rows,
        cols: cols,
        data: make([]float64, rows*cols),
    }
}

// --- Serialization for Matrix ---

// GobEncode allows Matrix to be saved despite having private fields.
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

// GobDecode allows Matrix to be loaded.
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

    return nil
}

// Randomize fills the matrix with random values (Standard Normal).
// We scale by sqrt(2/fanIn) (He Initialization) for better convergence.
func (m *Matrix) Randomize() {
    scale := math.Sqrt(2.0 / float64(m.rows))
    for i := range m.data {
        m.data[i] = rand.NormFloat64() * scale
    }
}

// NewMatrixFromSlice creates a Matrix view over an existing slice (Zero Copy).
// CAUTION: The underlying slice is shared.
func NewMatrixFromSlice(rows, cols int, data []float64) *Matrix {
    if len(data) != rows*cols {
        panic("Slice length does not match matrix dimensions")
    }
    return &Matrix{
        rows: rows,
        cols: cols,
        data: data,
    }
}

// Reset sets all values to 0.0 without re-allocating memory.
// This is much faster than creating a new Matrix.
func (m *Matrix) Reset() {
    for i := range m.data {
        m.data[i] = 0.0
    }
}

// MatMul performs C = A dot B, writing the result into 'out'.
// 'out' must be pre-allocated with correct shape.
func MatMul(a, b, out *Matrix) {
    if a.cols != b.rows || out.rows != a.rows || out.cols != b.cols {
        panic("Shape mismatch in MatMulInto")
    }

    // Reset output to clean state
    out.Reset()

    // Standard O(N^3) multiplication
    for i := 0; i < a.rows; i++ {
        for k := 0; k < a.cols; k++ {
            scalar := a.data[i*a.cols+k]
            // Vectorize the inner loop slightly by keeping index math simple
            rowOffsetC := i * out.cols
            rowOffsetB := k * b.cols
            for j := 0; j < b.cols; j++ {
                out.data[rowOffsetC+j] += scalar * b.data[rowOffsetB+j]
            }
        }
    }
}

// flatten converts a 2D slice into a 1D slice.
// It pre-allocates the exact memory needed to avoid resizing overhead.
func flatten(input [][]float64) []float64 {
    if len(input) == 0 {
        return nil
    }

    rows := len(input)
    cols := len(input[0])

    // 1. Allocate a single contiguous block of memory
    flat := make([]float64, rows*cols)

    // 2. Copy data row by row
    // Using 'copy' is generally faster than a manual loop because
    // it compiles down to a low-level memmove.
    for i, row := range input {
        copy(flat[i*cols:], row)
    }

    return flat
}

// AddVector adds a bias vector (row) to every row of the matrix.
func (m *Matrix) AddVector(v *Matrix) {
    for i := 0; i < m.rows; i++ {
        for j := 0; j < m.cols; j++ {
            m.data[i*m.cols+j] += v.data[j]
        }
    }
}

// Transpose returns the transpose of the matrix.
func (m *Matrix) Transpose() *Matrix {
    t := NewMatrix(m.cols, m.rows)
    for i := 0; i < m.rows; i++ {
        for j := 0; j < m.cols; j++ {
            t.data[j*t.cols+i] = m.data[i*m.cols+j]
        }
    }
    return t
}

// Subtract subtracts matrix b from a.
func (m *Matrix) Subtract(b *Matrix) {
    for i := range m.data {
        m.data[i] -= b.data[i]
    }
}

// ApplyFunc applies a function to every element.
func (m *Matrix) ApplyFunc(fn func(float64) float64) {
    for i := range m.data {
        m.data[i] = fn(m.data[i])
    }
}

// --- Activation Functions ---

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
        // Find max for numerical stability
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

// Helper to add matrices in place (needed for aggregation)
func (m *Matrix) Add(b *Matrix) {
    for i := range m.data {
        m.data[i] += b.data[i]
    }
}

// TransposeInto transposes 'm' into 'dst'. 
// 'dst' must be pre-allocated with dimensions (m.cols x m.rows).
func (m *Matrix) TransposeInto(dst *Matrix) {
    // Safety check (can be removed in production for speed)
    if dst.rows != m.cols || dst.cols != m.rows {
        panic(fmt.Sprintf("Shape mismatch in TransposeInto: Src(%d,%d) vs Dst(%d,%d)", 
            m.rows, m.cols, dst.rows, dst.cols))
    }

    // Standard cache-unfriendly loop (hard to optimize without blocking)
    for i := 0; i < m.rows; i++ {
        rowOffset := i * m.cols
        for j := 0; j < m.cols; j++ {
            dst.data[j*dst.cols+i] = m.data[rowOffset+j]
        }
    }
}

// --- Refactored Neural Network ---

type Layer struct {
    // Parameters (Shared among workers)
    Weights *Matrix
    Biases  *Matrix

    // Forward State (Unique to worker)
    Z *Matrix
    A *Matrix

    // Backward Scratchpads (Unique to worker) 
    // We pre-allocate these to avoid allocating them 60,000 times!
    dZ       *Matrix // Error signal for this layer
    AT       *Matrix // Transposed A (for dW calc)
    WeightsT *Matrix // Transposed Weights (for previous dZ calc)
}

type NeuralNetwork struct {
    Layers       []*Layer
    LearningRate float64
}

// CloneStructure creates a "shallow copy" for a worker.
// It shares the heavy Weights/Biases pointers but allocates fresh Z/A slots.
func (nn *NeuralNetwork) CloneStructure() *NeuralNetwork {
    newNN := &NeuralNetwork{
        LearningRate: nn.LearningRate,
        Layers:       make([]*Layer, len(nn.Layers)),
    }
    for i, l := range nn.Layers {
        newNN.Layers[i] = &Layer{
            Weights: l.Weights, // Share pointer!
            Biases:  l.Biases,  // Share pointer!
            // Z and A remain nil until Forward is called by the worker
        }
    }
    return newNN
}

func (nn *NeuralNetwork) InitializeBuffers(batchSize int) {
    for _, layer := range nn.Layers {
        // 1. Forward Buffers
        layer.Z = NewMatrix(batchSize, layer.Weights.cols)
        layer.A = NewMatrix(batchSize, layer.Weights.cols)
        
        // 2. Backward Buffers
        // dZ has same shape as A (Batch x OutputDim)
        layer.dZ = NewMatrix(batchSize, layer.Weights.cols)
        
        // AT has shape (OutputDim x Batch)
        layer.AT = NewMatrix(layer.Weights.cols, batchSize)

        // WeightsT has shape (OutputDim x InputDim)
        layer.WeightsT = NewMatrix(layer.Weights.cols, layer.Weights.rows)
    }
}

// Optimized Forward: Uses MatMulInto to avoid allocs
func (nn *NeuralNetwork) Forward(input *Matrix) {
    activation := input
    for i, layer := range nn.Layers {
        // reuse layer.Z memory
        MatMul(activation, layer.Weights, layer.Z)

        layer.Z.AddVector(layer.Biases)

        // reuse layer.A memory
        // Direct copy of Z to A before activation
        copy(layer.A.data, layer.Z.data)

        if i == len(nn.Layers)-1 {
            SoftmaxRow(layer.A)
        } else {
            layer.A.ApplyFunc(Relu)
        }
        activation = layer.A
    }
}

func (nn *NeuralNetwork) ComputeGradients(input *Matrix, Y []float64, grads []GradientSet) (float64, float64) {
    loss, acc := nn.ComputeLossAndAccuracy(Y)
    
    batchSize := float64(input.rows)
    scale := 1.0 / batchSize

    // --- 1. Calculate Output Error (Last Layer) ---
    lastLayerIdx := len(nn.Layers) - 1
    lastLayer := nn.Layers[lastLayerIdx]
    
    // We write directly into the last layer's dZ buffer
    // Start by copying A into dZ (dZ = A - Y)
    copy(lastLayer.dZ.data, lastLayer.A.data)
    
    // Subtract 1.0 from the correct class indices (Softmax/CrossEntropy derivative)
    for i, classLabel := range Y {
        // dZ[i][label] -= 1.0
        idx := i*lastLayer.dZ.cols + int(classLabel)
        lastLayer.dZ.data[idx] -= 1.0
    }

    // --- 2. Backpropagation Loop ---
    for i := lastLayerIdx; i >= 0; i-- {
        layer := nn.Layers[i]
        
        // A. Identify Previous Activation (A_prev)
        // If we are at the first layer, the "previous" activation is the Input.
        var A_prev *Matrix
        if i == 0 {
            A_prev = input 
            // Note: If 'input' is huge, transposing it is costly. 
            // In a strict zero-alloc system, you might pass an 'inputT' buffer 
            // or have a dedicated 'InputLayer' struct. 
            // For now, we assume we might allocate for input transpose OR reuse a buffer if implemented.
        } else {
            A_prev = nn.Layers[i-1].A
        }

        // B. Calculate dW (Gradient of Weights)
        // Formula: dW = A_prev^T * dZ
        
        // Optimization: Transpose A_prev into a pre-allocated buffer
        // If i==0, we might need a special buffer for input transpose, 
        // but for hidden layers, we use the PREVIOUS layer's AT buffer.
        if i > 0 {
            prevLayer := nn.Layers[i-1]
            A_prev.TransposeInto(prevLayer.AT) // Zero Alloc
            MatMul(prevLayer.AT, layer.dZ, grads[i].dW)
        } else {
            // Special case for Input layer to avoid complex struct changes:
            // We just allocate transpose here. (To fix this, add 'InputAT' to NeuralNetwork struct)
            A_prev_T := A_prev.Transpose() 
            MatMul(A_prev_T, layer.dZ, grads[i].dW)
        }

        // C. Calculate db (Gradient of Biases)
        // Formula: db = sum(dZ, axis=0)
        grads[i].db.Reset() // Clear old gradients
        
        // Sum columns manually (faster than a generic function)
        dZData := layer.dZ.data
        dbData := grads[i].db.data
        cols := layer.dZ.cols
        
        for r := 0; r < layer.dZ.rows; r++ {
            rowOffset := r * cols
            for c := 0; c < cols; c++ {
                dbData[c] += dZData[rowOffset+c]
            }
        }

        // D. Scale Gradients (Average by batch size)
        grads[i].dW.ApplyFunc(func(v float64) float64 { return v * scale })
        grads[i].db.ApplyFunc(func(v float64) float64 { return v * scale })

        // E. Calculate dZ for the *Previous* Layer (if more layers exist)
        // Formula: dZ_prev = (dZ * W^T) * ReluDerivative(Z_prev)
        if i > 0 {
            prevLayer := nn.Layers[i-1]
            
            // 1. Transpose current weights into scratchpad
            layer.Weights.TransposeInto(layer.WeightsT) // Zero Alloc
            
            // 2. Matrix Multiply into the PREVIOUS layer's dZ buffer
            // prevLayer.dZ = layer.dZ * layer.WeightsT
            MatMul(layer.dZ, layer.WeightsT, prevLayer.dZ)
            
            // 3. Apply Derivative (Relu) in-place
            // element-wise multiply: prevLayer.dZ *= Relu'(prevLayer.Z)
            zData := prevLayer.Z.data
            dZPrevData := prevLayer.dZ.data
            
            for k := range dZPrevData {
                if zData[k] <= 0 {
                    dZPrevData[k] = 0 // Relu Derivative is 0 if Z <= 0
                }
                // If Z > 0, Derivative is 1, so dZ stays the same.
            }
        }
    }

    return loss, acc
}

// SaveToFile saves the neural network weights and biases to a file.
func (nn *NeuralNetwork) SaveToFile(filename string) error {
    file, err := os.Create(filename)
    if err != nil {
        return err
    }
    defer file.Close()

    encoder := gob.NewEncoder(file)

    // We create a simpler struct just for saving needed data
    type NetworkData struct {
        Layers       []*Layer
        LearningRate float64
    }

    data := NetworkData{
        Layers:       nn.Layers,
        LearningRate: nn.LearningRate,
    }

    return encoder.Encode(data)
}

// LoadFromFile loads weights from a file into the current network.
func (nn *NeuralNetwork) LoadFromFile(filename string) error {
    file, err := os.Open(filename)
    if err != nil {
        return err
    }
    defer file.Close()

    decoder := gob.NewDecoder(file)

    // Helper struct matching the Save structure
    type NetworkData struct {
        Layers       []*Layer
        LearningRate float64
    }

    var data NetworkData
    if err := decoder.Decode(&data); err != nil {
        return err
    }

    // Apply loaded data
    nn.Layers = data.Layers
    nn.LearningRate = data.LearningRate

    // Important: Re-initialize buffers if needed (though typically Load happens before Init)
    return nil
}

// GradientSet holds the calculated gradients for one layer
type GradientSet struct {
    dW *Matrix
    db *Matrix
}

// --- Neural Network ---

func NewNeuralNetwork(layerSizes []int, lr float64) *NeuralNetwork {
    nn := &NeuralNetwork{LearningRate: lr}

    // Initialize layers: 784 -> 64 -> 32 -> 16 -> 10
    for i := 0; i < len(layerSizes)-1; i++ {
        inputSize := layerSizes[i]
        outputSize := layerSizes[i+1]

        layer := &Layer{
            Weights: NewMatrix(inputSize, outputSize),
            Biases:  NewMatrix(1, outputSize),
        }
        layer.Weights.Randomize()
        // Biases initialized to 0

        nn.Layers = append(nn.Layers, layer)
    }
    return nn
}

func (nn *NeuralNetwork) ComputeLossAndAccuracy(Y []float64) (float64, float64) {
    output := nn.Layers[len(nn.Layers)-1].A
    totalLoss := 0.0
    correctCount := 0

    epsilon := 1e-15 // Prevent log(0)

    for i := 0; i < output.rows; i++ {
        // Get predicted class (argmax)
        maxProb := -1.0
        predictedClass := -1

        targetClass := Y[i]

        for j := 0; j < output.cols; j++ {
            prob := output.data[i*output.cols+j]
            if prob > maxProb {
                maxProb = prob
                predictedClass = j
            }

            // Cross Entropy Loss sum: -log(p_correct)
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

// UpdateWeights applies the final averaged gradients to the shared weights.
// aggregatedGrads: The sum of gradients from all workers.
// splitScale: The factor to average them (usually 1.0 / GOMAXPROCS).
func (nn *NeuralNetwork) UpdateWeights(aggregatedGrads []GradientSet, splitScale float64) {
    // effectiveLR = LearningRate * (1 / G)
    // We combine these multiplications to iterate through the arrays only once.
    effectiveLR := nn.LearningRate * splitScale

    for i, layer := range nn.Layers {
        gradW := aggregatedGrads[i].dW
        gradB := aggregatedGrads[i].db

        // Update Weights: W = W - (gradW * effectiveLR)
        for j := range layer.Weights.data {
            layer.Weights.data[j] -= gradW.data[j] * effectiveLR
        }

        // Update Biases: B = B - (gradB * effectiveLR)
        for j := range layer.Biases.data {
            layer.Biases.data[j] -= gradB.data[j] * effectiveLR
        }
    }
}

// Predict takes a flattened image (1D slice), passes it through the network,
// and returns the predicted Class (0-9) and the Confidence (probability).
func (nn *NeuralNetwork) Predict(inputData []float64) (int, float64) {
    // 1. Validation
    inputSize := nn.Layers[0].Weights.rows
    if len(inputData) != inputSize {
        panic(fmt.Sprintf("Input size mismatch. Expected %d, got %d", inputSize, len(inputData)))
    }

    // Check if buffers exist and match the batch size (1)
    if nn.Layers[0].Z == nil || nn.Layers[0].Z.rows != 1 {
        nn.InitializeBuffers(1)
    }

    // 2. Create Matrix View (1 Row, N Cols)
    // We treat the single image as a Batch of size 1.
    inputMat := NewMatrixFromSlice(1, inputSize, inputData)

    // 3. Run Forward Pass
    // This updates the Master NN's internal 'A' matrices.
    nn.Forward(inputMat)

    // 4. Read Output
    // The result is in the last layer's Activation matrix (A).
    lastLayer := nn.Layers[len(nn.Layers)-1]
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

func convertJpg1D(path string) ([]float64, error) {
    f, err := os.Open(path)
    if err != nil {
        return nil, err
    }
    defer f.Close()

    img, err := jpeg.Decode(f)
    if err != nil {
        return nil, err
    }

    bounds := img.Bounds()
    w, h := bounds.Dx(), bounds.Dy()

    out := make([]float64, 0, w*h)

    for y := range h {
        for x := range w {
            r, g, b, _ := img.At(x, y).RGBA()

            gray := 0.299*float64(r>>8) +
                0.587*float64(g>>8) +
                0.114*float64(b>>8)

            out = append(out, gray)
        }
    }

    return out, nil
}

// ShuffleData randomizes X and Y in the exact same order using Fisher-Yates.
// This ensures that Image[i] still corresponds to Label[i].
func ShuffleData(X [][]float64, Y []float64) {
    rand.Seed(time.Now().UnixNano()) // Ensure random seed is set

    rand.Shuffle(len(X), func(i, j int) {
        // Swap indices in X (Input)
        X[i], X[j] = X[j], X[i]

        // Swap indices in Y (Labels) - MUST match the swap above!
        Y[i], Y[j] = Y[j], Y[i]
    })
}

// FlattenInto copies the 2D slice data into an EXISTING 1D slice.
// This avoids allocating new memory (malloc) every epoch, which is huge for performance.
func FlattenInto(source [][]float64, destination []float64) {
    cols := len(source[0])

    // Bounds check (optional, but good for safety)
    if len(destination) != len(source)*cols {
        panic("Destination slice size does not match source dimensions")
    }

    // Fast memory copy
    for i, row := range source {
        offset := i * cols
        copy(destination[offset:], row)
    }
}

func main() {
    G := runtime.GOMAXPROCS(runtime.NumCPU())
    fmt.Printf("Running on %d cores (Mini-Batch Mode)\n", runtime.GOMAXPROCS(0))

    // --- 1. Load Data ---
    X_raw, Y_raw, err := data.LoadCSV("mnist_train.csv")
    if err != nil {
        panic("Failed to load data: " + err.Error())
    }
    data.MinMaxNormalize(X_raw)

    inputDim := len(X_raw[0])
    numSamples := len(X_raw)

    // Create the single massive backing array (Zero Copy)
    X_global := NewMatrixFromSlice(len(X_raw), len(X_raw[0]), flatten(X_raw))

    layerSizes := []int{784, 64, 32, 16, 10}
    masterNN := NewNeuralNetwork(layerSizes, 0.1) // Higher LR is usually safe for mini-batch

    // --- 1. Auto-Load Logic ---
    modelFile := "mnist_model.gob"
    if _, err := os.Stat(modelFile); err == nil {
        fmt.Println("Found existing model. Loading weights...")
        if err := masterNN.LoadFromFile(modelFile); err != nil {
            fmt.Printf("Failed to load model: %v. Starting from scratch.\n", err)
        } else {
            fmt.Println("Weights loaded successfully!")
        }
    } else {
        fmt.Println("No existing model found. Starting training from scratch.")
    }

    // --- 2. Graceful Shutdown (Ctrl+C) ---
    // Create a channel to listen for OS signals
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

    go func() {
        <-sigChan // Block here until Ctrl+C is pressed
        fmt.Println("\n\nInterrupt received! Saving model...")

        // Save the current state of MasterNN
        if err := masterNN.SaveToFile(modelFile); err != nil {
            fmt.Printf("Error saving model: %v\n", err)
        } else {
            fmt.Printf("Model saved to %s\n", modelFile)
        }

        fmt.Println("Exiting.")
        os.Exit(0)
    }()

    // --- 2. CONFIGURATION ---
    BatchSize := G * 16 // 126 //numSamples-10// Standard mini-batch size

    // Ensure BatchSize is divisible by G for simplicity in this example
    if BatchSize%G != 0 {
        panic("For this example, please ensure BatchSize is divisible by Core Count (G)")
    }

    localBatchSize := BatchSize / G // e.g., 128 / 8 = 16 samples per worker

    // --- 3. PRE-ALLOCATE WORKER MEMORY (Small Buffers) ---
    workers := make([]*NeuralNetwork, G)
    workerGrads := make([][]GradientSet, G)

    for i := 0; i < G; i++ {
        // A. Clone Structure
        workers[i] = masterNN.CloneStructure()

        // B. Allocate Forward Buffers (Z, A)
        // Crucial: We only allocate enough for the LOCAL mini-batch (small!)
        workers[i].InitializeBuffers(localBatchSize)

        // C. Allocate Gradient Buffers (Same size as weights)
        workerGrads[i] = make([]GradientSet, len(masterNN.Layers))
        for l := 0; l < len(masterNN.Layers); l++ {
            rows, cols := masterNN.Layers[l].Weights.rows, masterNN.Layers[l].Weights.cols
            workerGrads[i][l].dW = NewMatrix(rows, cols)
            workerGrads[i][l].db = NewMatrix(1, cols)
        }
    }

    // --- 4. REUSABLE MASTER GRADIENT BUFFER ---
    finalGrads := make([]GradientSet, len(masterNN.Layers))
    for l := 0; l < len(masterNN.Layers); l++ {
        rows, cols := masterNN.Layers[l].Weights.rows, masterNN.Layers[l].Weights.cols
        finalGrads[l].dW = NewMatrix(rows, cols)
        finalGrads[l].db = NewMatrix(1, cols)
    }

    epochs := 20
    start := time.Now()

    fmt.Println("Starting Training...")

    for epoch := 1; epoch <= epochs; epoch++ {

        // --- THE SHUFFLE STEP ---
        // Only run this if we are doing Mini-Batch.
        // (If Full Batch, shuffling is mathematically useless and wastes time)
        if BatchSize < numSamples {

            // 1. Randomize the raw slices in sync
            ShuffleData(X_raw, Y_raw)

            // 2. Update X_global with the new order
            // We write directly into X_global.data. No new memory is created!
            FlattenInto(X_raw, X_global.data)
        }

        var totalLoss float64
        var totalAcc float64 // 1. Add accumulator for epoch accuracy
        batchesProcessed := 0

        // --- INNER BATCH LOOP ---
        // We iterate through the data in jumps of BatchSize
        for batchStart := 0; batchStart+BatchSize <= numSamples; batchStart += BatchSize {

            var wg sync.WaitGroup
            var batchLoss float64
            var batchAcc float64 // 2. Add accumulator for batch accuracy
            var statsMu sync.Mutex

            wg.Add(G)

            // Dispatch Workers
            for i := 0; i < G; i++ {
                go func(id int) {
                    defer wg.Done()

                    // 1. CALCULATE INDICES (The "Sliding Window")
                    // Global start of this batch + worker's local offset
                    globalStart := batchStart + (id * localBatchSize)
                    globalEnd := globalStart + localBatchSize

                    // 2. CREATE TEMPORARY INPUT VIEW
                    // We slice the massive X_global data.
                    // This is extremely cheap (just creating a header).
                    sliceStart := globalStart * inputDim
                    sliceEnd := globalEnd * inputDim

                    // Note: We don't need a persistent workerInput array anymore,
                    // we create this lightweight view on the fly.
                    localInput := NewMatrixFromSlice(localBatchSize, inputDim, X_global.data[sliceStart:sliceEnd])

                    // 3. TARGET LABELS
                    localY := Y_raw[globalStart:globalEnd]

                    // 4. FORWARD & BACKWARD
                    workers[id].Forward(localInput)

                    // 3. Capture both Loss and Accuracy here
                    loss, acc := workers[id].ComputeGradients(localInput, localY, workerGrads[id])

                    statsMu.Lock()
                    batchLoss += loss
                    batchAcc += acc // 4. Accumulate worker accuracy
                    statsMu.Unlock()
                }(i)
            }
            wg.Wait()

            // --- AGGREGATION PHASE (Happens every batch!) ---

            // Clear previous
            for l := 0; l < len(finalGrads); l++ {
                finalGrads[l].dW.Reset()
                finalGrads[l].db.Reset()
            }

            // Sum
            for _, gSet := range workerGrads {
                for l := 0; l < len(finalGrads); l++ {
                    finalGrads[l].dW.Add(gSet[l].dW)
                    finalGrads[l].db.Add(gSet[l].db)
                }
            }

            // Average and Update IMMEDIATELLY
            scale := 1.0 / float64(G)
            masterNN.UpdateWeights(finalGrads, scale)

            totalLoss += batchLoss / float64(G)
            totalAcc += batchAcc / float64(G)
            batchesProcessed++
        }

        // 6. Calculate Final Averages
        avgLoss := totalLoss / float64(batchesProcessed)
        avgAcc := (totalAcc / float64(batchesProcessed)) * 100 // Convert to percentage

        fmt.Printf("Epoch %d | Loss: %.4f | Acc: %.2f%% | Time: %v\n", epoch, avgLoss, avgAcc, time.Since(start))

        // fmt.Printf("Epoch %d | Avg Loss: %.4f | Time: %v\n", epoch, totalLoss/float64(batchesProcessed), time.Since(start))
    }

    // ... Inference code remains the same ...
    // [After loop ends]
    masterNN.SaveToFile(modelFile) // Save on normal completion
    fmt.Printf("Total Time: %v\n", time.Since(start))

    fmt.Println("Training Complete. Running Inference...")

    // 1. Load Image
    imagePath := "test/1.jpg"
    pixelData, err := convertJpg1D(imagePath)
    if err != nil {
        panic(err)
    }

    // 2. Normalize (Crucial!)
    // Assuming training data was MinMaxNormalized (0.0 to 1.0).
    // If your JPG reader returns 0-255, you MUST divide by 255.0.
    for i := range pixelData {
        pixelData[i] = pixelData[i] / 255.0
    }

    // 3. Predict
    prediction, confidence := masterNN.Predict(pixelData)

    fmt.Printf("Image: %s\n", imagePath)
    fmt.Printf("Predicted Digit: %d\n", prediction)
    fmt.Printf("Confidence: %.2f%%\n", confidence*100)
}