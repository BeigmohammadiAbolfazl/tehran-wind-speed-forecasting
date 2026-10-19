# Tehran Wind Speed Forecasting Using LSTM and ERA5

## Overview

This project focuses on short-term wind speed forecasting and preliminary spatial wind resource analysis in Tehran and its surrounding regions, Iran.

Hourly ERA5 reanalysis data are used to analyze spatial variations in wind speed and develop a Long Short-Term Memory (LSTM) neural network for one-hour-ahead forecasting.

## Objectives

- Analyze wind speed variations across the study region.
- Identify 20 grid points with relatively high mean wind speed at 100 m.
- Develop an LSTM model for short-term wind speed forecasting.
- Compare LSTM performance with a persistence baseline.
- Investigate wind conditions at grid points near central Tehran.

## Dataset

- **Source:** ECMWF ERA5 reanalysis
- **Study period:** January 1 – October 5, 2025
- **Spatial grid:** 81 grid points
- **Wind heights:** 10 m and 100 m
- **Forecast horizon:** One hour ahead

Wind speed is calculated from the horizontal wind components:

\[
V = \sqrt{u^2 + v^2}
\]

Dataset information: [ECMWF ERA5](https://www.ecmwf.int/en/forecasts/datasets/era5-hourly-data-single-levels-1940-present)

## Methodology

1. Load and preprocess ERA5 data.
2. Calculate wind speed at 10 m and 100 m.
3. Analyze spatial wind speed statistics.
4. Select 20 locations with relatively high mean wind speed at 100 m.
5. Construct input sequences from the previous 24 hourly observations and geographic coordinates.
6. Train an LSTM model to predict wind speed one hour ahead.
7. Evaluate the model using MAE, RMSE, and R².
8. Compare LSTM predictions against a persistence baseline.
9. Analyze four grid points within approximately 30 km of central Tehran.

## Model Architecture

The LSTM model consists of:

- LSTM layer with 64 units
- LSTM layer with 32 units
- Dropout rate of 0.2
- Dense layer with 16 units
- Single continuous output

The dataset was split chronologically into training (70%), validation (15%), and test (15%) sets.

## Results

### Overall LSTM Performance

| Metric | Result |
|---|---:|
| MAE | 0.4689 m/s |
| RMSE | 0.6358 m/s |
| R² | 0.9110 |

### Comparison with Persistence

| Metric | LSTM | Persistence |
|---|---:|---:|
| MAE | 0.4689 m/s | 0.5561 m/s |
| RMSE | 0.6358 m/s | 0.7521 m/s |
| R² | 0.9110 | 0.8754 |

The LSTM model reduced MAE by approximately 15.69% and RMSE by approximately 15.47% relative to the persistence baseline on the evaluated test set.

### Grid Points Near Tehran

| Metric | LSTM |
|---|---:|
| MAE | 0.3058 m/s |
| RMSE | 0.4279 m/s |
| R² | 0.9167 |

The spatial analysis identified differences in mean wind speed between the four grid points near central Tehran and the selected points in the broader study region. These findings are preliminary and do not establish long-term wind farm feasibility.

## Repository Contents

The repository includes Python scripts, model evaluation results, CSV files, and visualizations.

## Limitations

- The study period covers less than one full year.
- ERA5 grid values are not direct on-site wind measurements.
- Mean wind speed alone is insufficient to establish the economic feasibility of a wind farm.
- Long-term observations, wind power density, terrain, land use, environmental constraints, and grid access require further investigation.

## References

Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society, 146*(730), 1999–2049. https://doi.org/10.1002/qj.3803

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
