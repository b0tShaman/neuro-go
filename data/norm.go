package data

import (
	"encoding/json"
	"errors"
	"math"
	"os"
)

//
// ──────────────────────────────────────────────────────────────
//   DATA STRUCTURES
// ──────────────────────────────────────────────────────────────
//

type Mode int

const (
	MinMax Mode = iota
	ZScore
)

type MinMaxStats struct {
	Min []float64 `json:"min"`
	Max []float64 `json:"max"`
}

type ZScoreStats struct {
	Mean []float64 `json:"mean"`
	Std  []float64 `json:"std"`
}

func init() {
	// Remove old stats files (fresh start)
	os.Remove("stats_minmax.json")
	os.Remove("stats_zscore.json")
}

//
// ──────────────────────────────────────────────────────────────
//   COMPUTE STATS DURING TRAINING
// ──────────────────────────────────────────────────────────────
//

func ComputeMinMaxStats(X [][]float64) MinMaxStats {
	rows := len(X)
	cols := len(X[0])

	minVals := make([]float64, cols)
	maxVals := make([]float64, cols)

	for j := range cols {
		minVals[j] = X[0][j]
		maxVals[j] = X[0][j]
	}

	for i := 1; i < rows; i++ {
		for j := range cols {
			if X[i][j] < minVals[j] {
				minVals[j] = X[i][j]
			}
			if X[i][j] > maxVals[j] {
				maxVals[j] = X[i][j]
			}
		}
	}

	return MinMaxStats{Min: minVals, Max: maxVals}
}

func ComputeZScoreStats(X [][]float64) ZScoreStats {
	rows := len(X)
	cols := len(X[0])

	means := make([]float64, cols)
	stds := make([]float64, cols)

	// mean
	for i := range rows {
		for j := range cols {
			means[j] += X[i][j]
		}
	}
	for j := range cols {
		means[j] /= float64(rows)
	}

	// std
	for i := range rows {
		for j := range cols {
			d := X[i][j] - means[j]
			stds[j] += d * d
		}
	}
	for j := range cols {
		stds[j] = math.Sqrt(stds[j] / float64(rows))
	}

	return ZScoreStats{Mean: means, Std: stds}
}

//
// ──────────────────────────────────────────────────────────────
//   SAVE / LOAD STATS TO JSON
// ──────────────────────────────────────────────────────────────
//

func SaveStats(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func LoadStats(path string, v any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, v)
}

func MinMaxNormalize(X [][]float64) error {
	if len(X) == 0 || len(X[0]) == 0 {
		return errors.New("empty input")
	}

	statsFile := "stats_minmax.json"

	// Check if stats file exists → inference mode
	_, err := os.Stat(statsFile)
	if os.IsNotExist(err) {
		// ---------------- TRAINING MODE ----------------
		stats := ComputeMinMaxStats(X)

		if err := SaveStats(statsFile, stats); err != nil {
			return err
		}

		// Normalize using training stats
		return applyMinMaxInPlace(X, stats)
	}

	// ---------------- INFERENCE MODE ----------------
	var stats MinMaxStats
	if err := LoadStats(statsFile, &stats); err != nil {
		return err
	}

	return applyMinMaxInPlace(X, stats)
}

func NormalizeSampleMinMax(x []float64) error {
	var stats MinMaxStats

	if err := LoadStats("stats_minmax.json", &stats); err != nil {
		return err
	}
	return applyMinMax1D(x, stats)
}

func applyMinMaxInPlace(X [][]float64, stats MinMaxStats) error {
	min := stats.Min
	max := stats.Max

	rows := len(X)
	cols := len(X[0])

	for i := range rows {
		for j := range cols {
			den := max[j] - min[j]
			if den == 0 {
				X[i][j] = 0
			} else {
				X[i][j] = (X[i][j] - min[j]) / den
			}
		}
	}
	return nil
}

func applyMinMax1D(x []float64, stats MinMaxStats) error {
	min := stats.Min
	max := stats.Max

	if len(x) != len(min) {
		return errors.New("dimension mismatch")
	}

	for j := range x {
		den := max[j] - min[j]
		if den == 0 {
			x[j] = 0
		} else {
			x[j] = (x[j] - min[j]) / den
		}
	}
	return nil
}

func ZScoreNormalize(X [][]float64) error {
	if len(X) == 0 || len(X[0]) == 0 {
		return errors.New("empty input")
	}

	statsFile := "stats_zscore.json"

	_, err := os.Stat(statsFile)
	if os.IsNotExist(err) {
		// ---------------- TRAINING MODE ----------------
		stats := ComputeZScoreStats(X)

		if err := SaveStats(statsFile, stats); err != nil {
			return err
		}

		return applyZScoreInPlace(X, stats)
	}

	// ---------------- INFERENCE MODE ----------------
	var stats ZScoreStats
	if err := LoadStats(statsFile, &stats); err != nil {
		return err
	}

	return applyZScoreInPlace(X, stats)
}

func NormalizeSampleZScore(x []float64) error {
	var stats ZScoreStats

	if err := LoadStats("stats_zscore.json", &stats); err != nil {
		return err
	}
	return applyZScore1D(x, stats)
}

func applyZScoreInPlace(X [][]float64, stats ZScoreStats) error {
	mean := stats.Mean
	std := stats.Std

	rows := len(X)
	cols := len(X[0])

	for i := range rows {
		for j := range cols {
			if std[j] == 0 {
				X[i][j] = 0
			} else {
				X[i][j] = (X[i][j] - mean[j]) / std[j]
			}
		}
	}
	return nil
}

func applyZScore1D(x []float64, stats ZScoreStats) error {
	mean := stats.Mean
	std := stats.Std

	if len(x) != len(mean) {
		return errors.New("dimension mismatch")
	}

	for j := range x {
		if std[j] == 0 {
			x[j] = 0
		} else {
			x[j] = (x[j] - mean[j]) / std[j]
		}
	}
	return nil
}
