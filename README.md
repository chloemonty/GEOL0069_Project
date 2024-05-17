# GEOL0069_Project

<!-- DESCRIPTION -->
### Project Description

The goal for this project is to use artificial intelligence to train a model for sea ice roughness using RMS calculations for DEMs from drone-based photogrammetry data and HH backscatter Sentinel-1. The _project.ipynb_ notebook linked to this Github builds on the nmethods taught in the GEOL0069 Artificial Intelligence for Earth Observation module.

<!-- CONTEXT -->
### Context

<!-- GETTING STARTED -->
### Prerequisites

The following software needs to be installed to run the code.
* Mounting Google Drive on Google Colab
  ```sh
  from google.colab import drive
  drive.mount('/content/drive')
  ```
* Using pip install:
  ```sh
  !pip install GDAL
  ```
  ```sh
  !pip install numpy
  ```
  ```sh
  !pip install rasterio
  ```
  ```sh
  !pip install GPy
  ```

<!-- DATA -->
### Fetching Data

Define functions to get access tokens and download products by name from the Copernicus Data Space Ecosystem.

```sh
cop_dspace_usrnm = '' # amend with your copernicus dataspace username
cop_dspace_psswrd = ''# amend with your copernicus dataspace password
token, refresh_token = get_access_token(cop_dspace_usrnm, cop_dspace_psswrd)
product_names = [
    "S1A_EW_GRDM_1SDH_20230419T121123_20230419T121223_048165_05CA73_A681.SAFE",
    "S1A_EW_GRDM_1SDH_20230419T135049_20230419T135145_048166_05CA7C_5C05.SAFE"
]
```

## Pre-processing

The pre-processing of the Sentinel-1 data was carried out using the Sentinel Application Platform or SNAP by the European Space Agency. The steps included applying orbit files, thermal noise removal, calibration, a speckle-filter and ellipsoid correction. A subset was also defined over the area of interest.

## Converting .tif files to .npy data

A function is defined using GDAL to open the file, read the raster data as an array, calculate the X and Y coordinates for the associated pixel values, and export and save the X, Y and Z arrays, which in other words are the longitude, latitude and Sentinel-1 brightness values.

<!-- DEM -->
### Calculating RMS from the photogrammetry DEMs

Using a grid-size of 20 metres in order to have a few points per Sentinel-1 40x40m pixel

### Colocating Sentinel-1 and DEM data

Using KD-trees...

<!-- REGRESSION -->
### Testing different regression types

* Polynomial Regression:
* Neural Network Regression:
* Gaussian Process Regression with GPy:
* Gaussian Process Regression with GPSat:

<!-- POND INLET -->
### Using the polynomial model over sea ice
## Pond Inlet



## Cambridge Bay



## Other?



<!-- SENSITIVITY -->
### Sensitivity of predictions across the DEMs


