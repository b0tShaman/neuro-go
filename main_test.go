package main

import (
    "math/rand"
    "testing"
)

// --- 1. Helper: Naive Implementation for Comparison ---

// NaiveMatMul is the standard O(N^3) multiplication without cache blocking.
// We use this to prove your Tiled implementation is faster.
func NaiveMatMul(a, b, out *Matrix) {
    if a.cols != b.rows || out.rows != a.rows || out.cols != b.cols {
        panic("Shape mismatch in NaiveMatMul")
    }
    out.Reset()

    // Standard Triple Loop (Iterating row by row)
    for i := 0; i < a.rows; i++ {
        for k := 0; k < a.cols; k++ {
            scalar := a.data[i*a.cols+k]
            for j := 0; j < b.cols; j++ {
                out.data[i*out.cols+j] += scalar * b.data[k*b.cols+j]
            }
        }
    }
}

// --- 2. Benchmarks: Matrix Multiplication ---

var result *Matrix // Global variable to prevent compiler optimizations

func benchmarkMatMul(b *testing.B, size int, method string) {
    // Setup Matrices
    m1 := NewMatrix(size, size)
    m2 := NewMatrix(size, size)
    out := NewMatrix(size, size)
    
    m1.Randomize()
    m2.Randomize()

    b.ResetTimer()

    for n := 0; n < b.N; n++ {
        if method == "naive" {
            NaiveMatMul(m1, m2, out)
        } else {
            MatMul(m1, m2, out)
        }
    }
    result = out
}

func BenchmarkMatMul_Naive_64(b *testing.B)   { benchmarkMatMul(b, 64, "naive") }
func BenchmarkMatMul_Tiled_64(b *testing.B)   { benchmarkMatMul(b, 64, "tiled") }

func BenchmarkMatMul_Naive_256(b *testing.B)  { benchmarkMatMul(b, 256, "naive") }
func BenchmarkMatMul_Tiled_256(b *testing.B)  { benchmarkMatMul(b, 256, "tiled") }

func BenchmarkMatMul_Naive_512(b *testing.B)  { benchmarkMatMul(b, 512, "naive") }
func BenchmarkMatMul_Tiled_512(b *testing.B)  { benchmarkMatMul(b, 512, "tiled") }

func BenchmarkMatMul_Naive_1024(b *testing.B) { benchmarkMatMul(b, 1024, "naive") }
func BenchmarkMatMul_Tiled_1024(b *testing.B) { benchmarkMatMul(b, 1024, "tiled") }

// --- 3. Benchmarks: Full Network Inference ---

func benchmarkNetworkForward(b *testing.B, batchSize int) {
    // Setup Network (Standard MNIST Architecture)
	nn := NewNetwork(0.1,
		Input(784),
		Dense(64),
		Dense(32),
		Dense(16),
		Dense(10), // Will automatically be treated as Softmax
	)
    
    // Pre-allocate buffers for this batch size
    nn.InitializeBuffers(batchSize)

    // Create random input batch
    inputDim := 784
    inputData := make([]float64, batchSize*inputDim)
    for i := range inputData {
        inputData[i] = rand.Float64()
    }
    
    // Create Matrix View (Zero Copy)
    inputMat := NewMatrixFromSlice(batchSize, inputDim, inputData)

    b.ResetTimer()

    for n := 0; n < b.N; n++ {
        nn.Forward(inputMat)
    }
}

// Single item inference (Latency sensitive)
func BenchmarkForward_Batch_1(b *testing.B)   { benchmarkNetworkForward(b, 1) }

// Mini-batch inference (Throughput sensitive)
func BenchmarkForward_Batch_64(b *testing.B)  { benchmarkNetworkForward(b, 64) }
func BenchmarkForward_Batch_128(b *testing.B) { benchmarkNetworkForward(b, 128) }