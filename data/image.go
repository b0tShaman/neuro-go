package data

import (
	"image"
	"os"
	_ "image/jpeg" // Essential: Registers JPEG format
    _ "image/png"

	"golang.org/x/image/draw"
)

// Convert image of any size to grayscale 1D float64 slice
func ConvertJpg1D(path string, targetW, targetH int) ([]float64, error) {
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
