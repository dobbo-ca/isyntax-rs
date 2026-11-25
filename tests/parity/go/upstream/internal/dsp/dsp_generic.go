//go:build (!arm64 && !amd64) || purego

package dsp

func idwtRow(out, l, h []int16, _ *RowScratch) { idwtRowGo(out, l, h) }

var (
	liftOdd         = liftOddGo
	liftEven        = liftEvenGo
	YCoCgToRGBA     = ycocgGo
	YCoCgToRGB16    = ycocgRGB16Go
	PlanesToRGB16   = planesRGB16Go
	YCoCgToSamples  = ycocgSamplesGo
	PlanesToSamples = planesSamplesGo
	TruncRow        = truncGo
)
