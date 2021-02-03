package main

import "C"
import (
	"crypto/sha256"
	"encoding/base32"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path"
	"strconv"
	"time"

	"github.com/google/tiff"
	_ "github.com/google/tiff/geotiff"
	"github.com/google/tiff/image"
	"github.com/lukeroth/gdal"
	"github.com/sirupsen/logrus"
)

type options struct {
	logLevel string

	configPath string
}

func (o *options) validate() error {
	if o.configPath == "" {
		return errors.New("--config is required")
	}
	return nil
}

func bind() options {
	o := options{}
	fs := flag.NewFlagSet(os.Args[0], flag.ExitOnError)
	fs.StringVar(&o.logLevel, "log-level", "info", "Level at which to log output.")
	fs.StringVar(&o.configPath, "config", "", "Path to the configuration file to use.")
	if err := fs.Parse(os.Args[1:]); err != nil {
		logrus.WithError(err).Fatal("Could not parse flags.")
	}
	return o
}

type config struct {
	// Printer holds configuration about the printer used for this model.
	Printer printerConfig `json:"printer"`

	// Model holds configuration about the model.
	Model modelConfig `json:"model"`

	// Raster holds configuration for the raster we're working with.
	Raster rasterConfig `json:"raster"`
}

func (c *config) Validate() error {
	if err := c.Printer.Validate("printer"); err != nil {
		return err
	}
	if err := c.Model.Validate("model"); err != nil {
		return err
	}
	return nil
}

type printerConfig struct {
	// XYResolutionMicrons is the pixel resolution of the resin printer, in microns (μm).
	XYResolutionMicrons int `json:"xy_resolution_microns"`
	// ZResolutionMicrons is the expected Z layer height, in microns (μm).
	ZResolutionMicrons int `json:"z_resolution_microns"`

	// BedWidthMillimeters is the width of the printer's bed, in millimeters (mm).
	// This is optional and only used to optimize printing density.
	BedWidthMillimeters *int `json:"bed_width_millimeters,omitempty"`
	// BedLengthMillimeters is the width of the printer's bed, in millimeters (mm).
	// This is optional and only used to optimize printing density.
	BedLengthMillimeters *int `json:"bed_length_millimeters,omitempty"`
	// ZAxisHeightMillimeters is the height of the printer's Z axis, in millimeters (mm).
	// This is optional and only used to ensure all model parcels fit into the build volume.
	ZAxisHeightMillimeters *int `json:"z_axis_height_millimeters,omitempty"`
}

func (p *printerConfig) Validate(root string) error {
	if p.XYResolutionMicrons == 0 {
		return fmt.Errorf("%s.xy_resolution_microns: must be set", root)
	}
	if p.ZResolutionMicrons == 0 {
		return fmt.Errorf("%s.z_resolution_microns: must be set", root)
	}

	dimensionsProvided := 0
	if p.BedWidthMillimeters != nil {
		dimensionsProvided++
	}
	if p.BedLengthMillimeters != nil {
		dimensionsProvided++
	}
	if dimensionsProvided == 1 {
		return fmt.Errorf("%s.bed_{width,length}_millimeters: either both the bed width and length must be provided at oince, or neither, but not just one", root)
	}
	return nil
}

type modelConfig struct {
	// WidthMillimeters is the desired overall width of the finished model, in millimeters (mm).
	// Optional, the overall model size can be set explicitly with dimensions or by specifying
	// the scaling factor to apply to the optimal projection representation.
	WidthMillimeters *int `json:"width_millimeters,omitempty"`
	// LengthMillimeters is the desired overall length of the finished model, in millimeters (mm).
	// Optional, the overall model size can be set explicitly with dimensions or by specifying
	// the scaling factor to apply to the optimal projection representation.
	LengthMillimeters *int `json:"length_millimeters,omitempty"`

	// XYScale is the scaling factor to use for the XY projection, as compared to the optimal print
	// density where each pixel in the GeoTiff raster corresponds to one pixel in the printer's LCD.
	//Optional, model size can be set with this scaling factor or explicitly with dimensions.
	XYScale *float64 `json:"xy_scale,omitempty"`

	// ZScale is the scaling factor to use for the Z relief. Optional, defaults to 1.
	ZScale *float64 `json:"z_scale,omitempty"`

	// ParcelWidthMillimeters is the width of any individual parcel that will be printed. Parcels
	// are to be assembled after the fact into the final model. Optional, will default to something
	// sensible to maximize the printer's available bed.
	ParcelWidthMillimeters *int `json:"parcel_width_millimeters,omitempty"`
	// ParcelAspectRatio is the ratio of a parcel's width to it's height. Optional, defaults to 1.
	ParcelAspectRatio *float64 `json:"parcel_aspect_ratio,omitempty"`
}

func (m *modelConfig) Validate(root string) error {
	dimensionsProvided := 0
	if m.WidthMillimeters != nil {
		dimensionsProvided++
	}
	if m.LengthMillimeters != nil {
		dimensionsProvided++
	}
	if dimensionsProvided == 1 {
		return fmt.Errorf("%s.{width,length}_millimeters: either both the bed width and length must be provided at oince, or neither, but not just one", root)
	}
	if dimensionsProvided == 2 && m.XYScale != nil {
		return fmt.Errorf("%s.{width,length}_millimeters cannot be set at the same time as %s.xy_scale", root, root)
	}
	if dimensionsProvided == 0 && m.XYScale == nil {
		return fmt.Errorf("%s.{width,length}_millimeters or %s.xy_scale must be set", root, root)
	}
	return nil
}

type rasterConfig struct {
	// Path is the location of the GeoTiff raster to load.
	Path string `json:"path"`

	// Bounds are the bounds for values to use from the raster. If bounds are set, values in the
	// raster outside of them are set to NaN and not included in the output geometry.
	Bounds *bounds `json:"bounds,omitempty"`
}

func (r *rasterConfig) Validate(root string) error {
	if r.Path == "" {
		return fmt.Errorf("%s.path: must be set", root)
	}
	return nil
}

type bounds struct {
	// Lower is the lower bound for values to use from the raster. Optional.
	Lower *float64 `json:"lower,omitempty"`
	// Upper is the upper bound for values to use from the raster. Optional.
	Upper *float64 `json:"upper,omitempty"`
}

type rasterInfo struct {
	// pixelSize is the length, in meters, between pixels in X and Y directions
	pixelSize float64
	// xSize and ySize are the number of pixels in the raster's width and length
	xSize, ySize int
}

func main() {
	o := bind()
	if err := o.validate(); err != nil {
		logrus.WithError(err).Fatal("Invalid options.")
	}
	level, err := logrus.ParseLevel(o.logLevel)
	if err != nil {
		logrus.WithError(err).Fatal("invalid --log-level")
	}
	logrus.SetLevel(level)

	rawConfig, err := ioutil.ReadFile(o.configPath)
	if err != nil {
		logrus.WithError(err).Fatal("Could not load configuration.")
	}
	var conf config
	if err := json.Unmarshal(rawConfig, &conf); err != nil {
		logrus.WithError(err).Fatal("Could not unmarshal configuration.")
	}
	if err := conf.Validate(); err != nil {
		logrus.WithError(err).Fatal("Configuration is invalid.")
	}

	// We hash the input configuration to create a deterministic spot to store intermediate
	// files, logs, etc for this process and allow us to recover from crashes without re-doing
	// work.
	hash := sha256.New()
	if _, err := hash.Write(rawConfig); err != nil {
		logrus.WithError(err).Error("Failed to compute hash of input configuration.")
	}

	// We don't want this directory name to be too long so we truncate the hash. This increases
	// chances of collision but we can tolerate it as our input space is tiny.
	name := base32.StdEncoding.EncodeToString(hash.Sum(nil)[:15])
	home, err := os.UserHomeDir()
	if err != nil {
		logrus.WithError(err).Fatal("Could not determine user's home directory.")
	}
	cacheDir := path.Join(home, "terrain", name)
	logrus.Infof("Initializing cache to %s", cacheDir)
	if err := os.MkdirAll(cacheDir, os.ModePerm); err != nil {
		logrus.WithError(err).Fatalf("Could not initialize cache at %s.", cacheDir)
	}

	logrus.Info("Loading GeoTiff...")
	start := time.Now()
	dataset, err := gdal.Open(conf.Raster.Path, gdal.ReadOnly)
	if err != nil {
		logrus.WithError(err).Fatal("Could not read GeoTiff.")
	}
	logrus.WithField("duration", time.Since(start)).Debug("Loaded GeoTiff.")

	// The default GDAL load will only read the coordinate system for the projection, not any for
	// the vertical coordinates. However, the spec declares that the *same* coordinate system is
	// to be used for both. Therefore, while we only load and use the units for the X/Y projection,
	// we are OK to assume that they apply as well to the Z.
	spatialReference := gdal.CreateSpatialReference(dataset.Projection())
	rawUnit, found := spatialReference.AttrValue("UNIT", 1)
	if !found {
		logrus.Fatal("Could not find the units used in the projection coordinate system.")
	}
	unit, err := strconv.ParseFloat(rawUnit, 64) // in meters
	if err != nil {
		logrus.WithError(err).Fatal("Could not parse the projection coordinate system units.")
	}

	// In a north up image, transform[1] is the pixel width, transform [5] is the pixel height.
	// The upper left corner of the upper left pixel is at position (transform[0],transform[3]).
	transform := dataset.GeoTransform()
	pixelWidth, pixelHeight := math.Abs(transform[1])*unit, math.Abs(transform[5])*unit
	if pixelHeight != pixelWidth {
		logrus.Debugf("Pixel width (%fm) is not the same as pixel height (%fm). Will assume even grid spacing using width.", pixelWidth, pixelHeight)
	}
	data := dataset.RasterBand(1)
	info := rasterInfo{
		pixelSize: pixelWidth,
		xSize:     data.XSize(),
		ySize:     data.YSize(),
	}
	logrus.Infof("Source raster data has shape (%d, %d) and uses a %fm grid.", info.xSize, info.ySize, info.pixelSize)
	logrus.Debugf("Source raster upper left corner is at (%0.4f,%0.4f).", transform[0], transform[3])

	xSize, ySize, xFactor, yFactor := scaledRasterDimensions(conf.Printer, conf.Model, info)
	logrus.Infof("Re-sampling raster data with a scaling factor of (%0.2f,%0.2f) to (%d,%d)", xFactor, yFactor, xSize, ySize)

	//var buffer interface{}
	//switch data.RasterDataType() {
	//case gdal.UInt16:
	//	buffer = make([]uint16, info.xSize * info.ySize)
	//case gdal.Int16:
	//	buffer = make([]int16, info.xSize * info.ySize)
	//case gdal.UInt32:
	//	buffer = make([]uint32, info.xSize * info.ySize)
	//case gdal.Int32:
	//	buffer = make([]int32, info.xSize * info.ySize)
	//case gdal.Float32:
	//	buffer = make([]float32, info.xSize * info.ySize)
	//case gdal.Float64:
	//	buffer = make([]float64, info.xSize * info.ySize)
	//default:
	//	logrus.Fatalf("Unsupported data format in raster band: %v", data.RasterDataType().Name())
	//}
	//logrus.Debugf("Reading raster band data into type %v", data.RasterDataType().Name())
	//if err := data.IO(gdal.Read, 0, 0, info.xSize, info.ySize, buffer, info.xSize, info.ySize, 0, 0); err != nil {
	//	logrus.WithError(err).Fatal("Failed to read raster band.")
	//}

	file, err := os.Open(conf.Raster.Path)
	if err != nil {
		logrus.WithError(err).Fatal("Could not open GeoTiff file.")
	}
	defer func() {
		if err := file.Close(); err != nil {
			logrus.WithError(err).Error("Could not close GeoTiff file.")
		}
	}()
	tiffData, err := tiff.Parse(file, nil, nil)
	if err != nil {
		logrus.WithError(err).Fatal("Could not load GeoTiff.")
	}
	for _, field := range tiffData.IFDs()[0].Fields() {
		logrus.Info(field)
	}
	img, err := image.Decode(file)
	if err != nil {
		logrus.WithError(err).Fatal("Could not parse GeoTiff raster band as image.")
	}
	logrus.Info(img.Bounds())
}

// scaledRasterDimensions determines the scaling factor which which we need to re-process the input raster
// in order to make sure that it has the same level of detail as the user has requested in the
// final model given their printer's capabilities.
func scaledRasterDimensions(printer printerConfig, model modelConfig, info rasterInfo) (int, int, float64, float64) {
	xyResolutionMeters := float64(printer.XYResolutionMicrons) / float64(1e6)
	var xSize, ySize int
	var xScale, yScale float64
	switch {
	case model.LengthMillimeters != nil && model.WidthMillimeters != nil:
		millimetersToMeters := func(mm int) float64 {
			return float64(mm) / float64(1e3)
		}
		modelXPixelCount, modelYPixelCount := millimetersToMeters(*model.WidthMillimeters)/xyResolutionMeters, millimetersToMeters(*model.LengthMillimeters)/xyResolutionMeters
		logrus.Debugf("With a resolution of %dμm and a model size of %dmm x %dmm, model will be %0.2f x %0.2f pixels.", printer.XYResolutionMicrons, model.WidthMillimeters, model.LengthMillimeters, modelXPixelCount, modelYPixelCount)
		xScale, yScale = modelXPixelCount/float64(info.xSize), modelYPixelCount/float64(info.ySize)
		logrus.Infof("To achieve the requested model size, the input raster will need to be scaled by %0.2f in X and %0.2f in Y", xScale, yScale)
		if xScale > 1.0 || yScale > 1.0 {
			logrus.Warn("In achieving the requested model size, the input raster will need to be over-sampled. Consider reducing the model size to match the native resolution of your printer.")
		}
		xSize, ySize = int(math.Round(modelXPixelCount)), int(math.Round(modelYPixelCount))
	case model.XYScale != nil:
		pixelsToMillimeters := func(pixels int) float64 {
			return float64(pixels) * xyResolutionMeters * float64(1e3)
		}
		logrus.Debugf("With a size of %d x %d pixels, the raster would have a size of %0.2fmm x %0.2fmm at the printer's native resolution of resolution of %dμm.", info.xSize, info.ySize, pixelsToMillimeters(info.xSize), pixelsToMillimeters(info.ySize), printer.XYResolutionMicrons)
		modelXPixelCount, modelYPixelCount := int(math.Round(float64(info.xSize)**model.XYScale)), int(math.Round(float64(info.ySize)**model.XYScale))
		logrus.Infof("With a size of %d x %d pixels, the raster will have a size of %0.2fmm x %0.2fmm using a scaling factor of %0.2f.", modelXPixelCount, modelYPixelCount, pixelsToMillimeters(modelXPixelCount), pixelsToMillimeters(modelXPixelCount), *model.XYScale)
		xSize, ySize, xScale, yScale = modelXPixelCount, modelYPixelCount, *model.XYScale, *model.XYScale
	}
	logrus.Infof("Final model scale will be 1:%0.2f", info.pixelSize/(xScale*xyResolutionMeters))
	return xSize, ySize, xScale, yScale
}
