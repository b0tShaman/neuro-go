package data

import (
	"encoding/csv"
	"fmt"
	"os"
	"sort"
	"strconv"
)

// LoadCSV reads a CSV file and returns numeric features X and target Y.
// It automatically converts categorical (string) features into one-hot encoded vectors.
// It assumes the last column is the target variable.
// It skips the header row.
func LoadCSV(filename string) ([][]float64, []float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	if len(records) < 2 {
		return nil, nil, fmt.Errorf("not enough rows in CSV")
	}

	numCols := len(records[0])

	// --- Detect categorical columns ---
	isCategorical := make([]bool, numCols-1) // exclude last column (Y)
	categoryValues := make([]map[string]struct{}, numCols-1)

	for i := 0; i < numCols-1; i++ {
		categoryValues[i] = make(map[string]struct{})
		isCategorical[i] = false
	}

	// Detect categorical by checking if ParseFloat fails
	for _, row := range records[1:] {
		for j := 0; j < numCols-1; j++ {
			_, err := strconv.ParseFloat(row[j], 64)
			if err != nil {
				isCategorical[j] = true
				categoryValues[j][row[j]] = struct{}{}
			}
		}
	}

	// --- Create consistent ordering of categories for one-hot encoding ---
	categoryOrder := make([][]string, numCols-1)
	for j := 0; j < numCols-1; j++ {
		if isCategorical[j] {
			for val := range categoryValues[j] {
				categoryOrder[j] = append(categoryOrder[j], val)
			}
			sort.Strings(categoryOrder[j]) // deterministic order
		}
	}

	// --- Build feature matrix X and target vector Y ---
	var X [][]float64
	var Y []float64

	for _, row := range records[1:] {
		var features []float64

		for j := 0; j < numCols-1; j++ {
			if isCategorical[j] {
				oneHot := make([]float64, len(categoryOrder[j]))
				val := row[j]
				for k, cat := range categoryOrder[j] {
					if val == cat {
						oneHot[k] = 1.0
						break
					}
				}
				features = append(features, oneHot...)
			} else {
				val, err := strconv.ParseFloat(row[j], 64)
				if err != nil {
					return nil, nil, fmt.Errorf("invalid numeric value %q in column %d", row[j], j)
				}
				features = append(features, val)
			}
		}

		y, err := strconv.ParseFloat(row[numCols-1], 64)
		if err != nil {
			return nil, nil, fmt.Errorf("invalid target value %q", row[numCols-1])
		}

		X = append(X, features)
		Y = append(Y, y)
	}

	return X, Y, nil
}

func LoadCSVSeq(filename string) ([][]float64, [][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	rows, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	if len(rows) == 0 {
		return nil, nil, nil
	}

	// Infer contextLen: total columns = 2*contextLen
	numCols := len(rows[0])
	if numCols%2 != 0 {
		return nil, nil, fmt.Errorf("invalid dataset row width %d, expected even number", numCols)
	}

	contextLen := numCols / 2

	var X [][]float64
	var Y [][]float64

	for _, row := range rows {

		if len(row) != numCols {
			return nil, nil, fmt.Errorf("row length mismatch: expected %d got %d", numCols, len(row))
		}

		xVals := make([]float64, contextLen)
		yVals := make([]float64, contextLen)

		for i := 0; i < contextLen; i++ {
			v, err := strconv.ParseFloat(row[i], 64)
			if err != nil {
				return nil, nil, err
			}
			xVals[i] = v
		}

		for i := 0; i < contextLen; i++ {
			v, err := strconv.ParseFloat(row[i+contextLen], 64)
			if err != nil {
				return nil, nil, err
			}
			yVals[i] = v
		}

		X = append(X, xVals)
		Y = append(Y, yVals)
	}

	return X, Y, nil
}
