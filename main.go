package main

import (
    "bytes"
    "encoding/gob"
    "fmt"
    "image"
    "math"
    "math/rand"
    "os"
    "os/signal"
    "runtime"
    "sync"
    "syscall"
    "time"

    "github.com/b0tShaman/go-rout-net/data"
    "golang.org/x/image/draw"
    "gonum.org/v1/gonum/floats"
    "gonum.org/v1/gonum/mat"
)

const (
    ActLinear ActivationType = iota
    ActRelu
    ActSigmoid
    ActSoftmax
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

    // Track the output size of the previous layer
    prevOutputSize := configs[0].Neurons

    for i := 1; i < len(configs); i++ {
        cfg := configs[i]
        act := cfg.Activation

        layer := &Layer{
            Weights: NewMatrix(prevOutputSize, cfg.Neurons),
            Biases:  NewMatrix(1, cfg.Neurons),
            ActType: act,
        }

        layer.Weights.Randomize()
        nn.Layers = append(nn.Layers, layer)

        // Update for next iteration
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

// -------- MAIN -------- //
func main() {
    // Hardware Setup
    G := runtime.GOMAXPROCS(runtime.NumCPU())
    modelFile := "assets/model.gob"

    // 1. Load Data
    fmt.Println("Loading dataset...")
    X_raw, Y_raw, err := data.LoadCSV("assets/mnist_train.csv")
    if err != nil {
        panic("Failed to load data: " + err.Error())
    }

    // Normalize Input Data
    for i := range X_raw {
        for j := range X_raw[i] {
            X_raw[i][j] /= 255.0
        }
    }

    // Create Global Matrix (Zero Copy)
    X_global := NewMatrixFromSlice(len(X_raw), len(X_raw[0]), flatten(X_raw))
    inputDim := len(X_raw[0])

    // 2. Initialize Network
    nw := NewNetwork(
        Input(inputDim),
        Dense(64),
        Dense(32),
        Dense(16),
        Dense(10, Activation("softmax")),
    )

    // Auto-Load weights if they exist
    if _, err := os.Stat(modelFile); err == nil {
        fmt.Println("Found existing model. Loading weights...")
        nw.LoadFromFile(modelFile)
    }

    // 3. Configure & Train
    config := TrainingConfig{
        Epochs:       20,
        BatchSize:    G * 16,
        LearningRate: 0.1, // Adam default is 0.001, SGD can be higher
        ModelPath:    modelFile,
        NumWorkers:   G,
        Optimizer:    OptSGD,
        MomentumMu:   0.9, // Only used if OptMomentum is selected
    }

    fmt.Printf("Running on %d cores (Mini-Batch = %d)\n\n", runtime.GOMAXPROCS(0), config.BatchSize)

    // Run Training
    nw.Train(X_global, Y_raw, config)

    // 4. Run Inference
    nw.InferenceImg("assets/5.jpg")
}

// -------- NEURAL NETWORK METHODS -------- //
func (nw *NeuralNetwork) Train(X_global *Matrix, Y_raw []float64, cfg TrainingConfig) {
    fmt.Printf("TrainingConfig: %+v\n", cfg)

    if cfg.BatchSize%cfg.NumWorkers != 0 {
        panic("BatchSize must be divisible by NumWorkers")
    }

    // Create the chosen optimizer
    optimizer := NewOptimizer(nw, cfg)

    localBatchSize := cfg.BatchSize / cfg.NumWorkers
    inputDim := X_global.cols
    numSamples := X_global.rows

    // 1. PRE-ALLOCATE WORKER MEMORY
    workers := make([]*NeuralNetwork, cfg.NumWorkers)
    workerGrads := make([][]GradientSet, cfg.NumWorkers)

    fmt.Printf("Initializing %d workers (Local Batch: %d)\n", cfg.NumWorkers, localBatchSize)

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
    data := make([]float64, batchSize*inputDim)
    nw.InputBuf = &Matrix{
        rows:  batchSize,
        cols:  inputDim,
        data:  data,
        dense: mat.NewDense(batchSize, inputDim, data),
    }

    nw.InputT = NewMatrix(inputDim, batchSize)

    for _, layer := range nw.Layers {
        layer.Z = NewMatrix(batchSize, layer.Weights.cols)
        layer.A = NewMatrix(batchSize, layer.Weights.cols)
        layer.dZ = NewMatrix(batchSize, layer.Weights.cols)
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

// Optimized Forward with Activation Switch
func (nw *NeuralNetwork) Forward(input *Matrix) {
    activation := input
    for _, layer := range nw.Layers {
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

func (nw *NeuralNetwork) ComputeGradients(input *Matrix, Y []float64, grads []GradientSet) (float64, float64) {
    loss, acc := nw.ComputeLossAndAccuracy(Y)

    batchSize := float64(input.rows)
    scale := 1.0 / batchSize

    lastLayerIdx := len(nw.Layers) - 1
    lastLayer := nw.Layers[lastLayerIdx]

    // 1. Output Error (Softmax + CrossEntropy)
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

        // Determine A_prev (The input to this layer)
        var A_prev_dense mat.Matrix // Interface type
        if i == 0 {
            A_prev_dense = input.dense
        } else {
            A_prev_dense = nw.Layers[i-1].A.dense
        }

        // --- CALC dW ---
        if i > 0 {
            // dW = A_prev.T * dZ
            MatMul(A_prev_dense.T(), layer.dZ.dense, grads[i].dW)
        } else {
            // Special case for input layer
            MatMul(input.dense.T(), layer.dZ.dense, grads[i].dW)
        }

        // --- CALC db ---
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

        // Apply Scale
        grads[i].dW.ApplyFunc(func(v float64) float64 { return v * scale })
        grads[i].db.ApplyFunc(func(v float64) float64 { return v * scale })

        // floats.Scale(scale, grads[i].dW.data) -> this would be faster
        // floats.Scale(scale, grads[i].db.data)

        // --- CALC dZ_prev ---
        if i > 0 {
            prevLayer := nw.Layers[i-1]
            MatMul(layer.dZ.dense, layer.Weights.dense.T(), prevLayer.dZ)

            // Apply Derivative
            zData := prevLayer.Z.data
            dZPrevData := prevLayer.dZ.data

            // Note: Since we use slice access for element-wise ops,
            // we don't need Gonum here. This part stays fast & custom.
            for k := range dZPrevData {
                switch prevLayer.ActType {
                case ActRelu:
                    if zData[k] <= 0 {
                        dZPrevData[k] = 0
                    }
                case ActSigmoid:
                    a := prevLayer.A.data[k]
                    dZPrevData[k] *= a * (1.0 - a)
                case ActLinear:
                    // derivative is 1
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

    type LayerData struct {
        Weights *Matrix
        Biases  *Matrix
        ActType ActivationType
    }
    type NetworkData struct {
        LayerDatas   []LayerData
        LearningRate float64
    }

    var data NetworkData
    if err := decoder.Decode(&data); err != nil {
        return err
    }

    nw.Layers = make([]*Layer, len(data.LayerDatas))
    for i, ld := range data.LayerDatas {
        nw.Layers[i] = &Layer{
            Weights: ld.Weights,
            Biases:  ld.Biases,
            ActType: ld.ActType,
        }
    }
    nw.LearningRate = data.LearningRate
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
// and returns the predicted Class (0-9) and the Confidence (probability).
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