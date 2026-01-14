# Example Weather Data for maisUI

This directory contains example weather CSV files for testing the maisUI interface.

## Files

- `example_pre_seedling.csv`: 30 days of pre-planting weather from trial 439 (2022)
- `example_growing_season.csv`: Growing season weather from the same trial (154 days)

## CSV Format

All files have the following structure:
- `date`: Date (YYYY-MM-DD)
- 32 weather feature columns matching the preprocessor

Weather features include: potential_evaporation, soil_temperature_level_1, soil_temperature_level_2, surface_net_solar_radiation, surface_net_thermal_radiation... (and 27 more)

## Usage in maisUI

1. Start maisUI: `cd ../maisUI && python app.py`
2. In the weather section, upload:
   - Pre-seedling CSV (30 days before planting)
   - Growing season CSV (planting to harvest, up to 200 days)

## Source

- Database: data/cornN.duckdb
- Trial: 439 (2022)
- Location: 46.61°N, -71.17°W
- Extraction date: 2026-01-12 20:32
