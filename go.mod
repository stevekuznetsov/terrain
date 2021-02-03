module github.com/stevekuznetsov/terrain

go 1.15

replace github.com/lukeroth/gdal v0.0.0-20210115064924-635e469daf3b => github.com/tingold/gdal-1 v0.0.0-20200805034744-092f31c3aae1

require (
	github.com/google/tiff v0.0.0-20161109161721-4b31f3041d9a
	github.com/lukeroth/gdal v0.0.0-20210115064924-635e469daf3b
	github.com/sirupsen/logrus v1.7.0
)
