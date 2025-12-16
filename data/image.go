package data

import (
	"image/jpeg"
	"os"
)

// convertJpg1D reads an image file, converts to grayscale if needed,
// flattens it row-major, and returns []float64
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
