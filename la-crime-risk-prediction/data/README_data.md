# Data Sources

Raw data files are not included in this repository.  
Follow the instructions below to download each dataset and place it in this `data/` folder.

---

## 1. LAPD Crime Data (primary dataset)

**Source:** City of Los Angeles Open Data Portal  
**URL:** https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-Present/2nrs-mtv8

Download steps:
1. Go to the URL above
2. Click **Export** → **CSV**
3. Save as `data/lapd_crime.csv`

Dataset coverage: 2020–present, 1,000,000+ rows, 28 columns  
Includes: crime type, date/time, location (lat/lon), victim demographics, premise type

---

## 2. American Community Survey (ACS) — Census Demographics

**Source:** Google BigQuery Public Datasets  
**Dataset:** `bigquery-public-data.census_bureau_acs`

This project uses block-group level demographic features including median income,
racial composition, housing tenure, unemployment rate, and rent burden.

Query example (run in BigQuery console or via `pandas-gbq`):

```sql
SELECT
  geo_id,
  total_pop,
  median_income,
  income_per_capita,
  white_pop,
  black_pop,
  asian_pop,
  hispanic_pop,
  housing_units,
  occupied_housing_units,
  housing_units_renter_occupied,
  owner_occupied_housing_units,
  median_rent,
  households,
  pop_in_labor_force,
  unemployed_pop,
  commuters_16_over,
  rent_over_50_percent
FROM
  `bigquery-public-data.census_bureau_acs.blockgroup_2020_5yr`
```

Save the result as `data/acs_blockgroup.csv`

Requires a Google Cloud account (free tier is sufficient for this query).

---

## 3. LA County COVID-19 Cases

**Source:** LA County Open Data  
**URL:** https://data.lacounty.gov/datasets/lacounty::covid-19-la-county-cases/about

Download steps:
1. Go to the URL above
2. Click **Download** → **CSV**
3. Save as `data/la_covid_cases.csv`

---

## 4. Weather Data — NOAA GHCN

**Source:** NOAA Global Historical Climatology Network  
**URL:** https://www.ncdc.noaa.gov/cdo-web/

Download steps:
1. Go to the URL above and click **Order Data**
2. Select dataset: **Daily Summaries**
3. Search for station: **USC00045114** (Los Angeles Civic Center) or nearest available
4. Date range: **2020-01-01 to 2023-12-31**
5. Select fields: `TAVG`, `TMAX`, `TMIN`, `PRCP`
6. Output format: **CSV**
7. Save as `data/noaa_weather.csv`

---

## 5. Geocoding (pre-processed, no download needed)

Crime records without lat/lon coordinates were geocoded using a two-stage pipeline:
- **Primary:** US Census Geocoder API — https://geocoding.geo.census.gov/geocoder/
- **Fallback:** OpenStreetMap Nominatim — https://nominatim.openstreetmap.org/

This process achieved 92% coverage (924k / 1M rows) and is already reflected
in the merged dataset `crime_with_acs_and_more.csv`.  
If you are reproducing from scratch, see the geocoding logic in `src/preprocess.py`.

---

## Final merged file

Once all sources are downloaded, the pipeline expects a single merged CSV:

```
data/crime_with_acs_and_more.csv
```

This file is produced by joining the LAPD crime data with ACS block group demographics
(via Census GEOID), COVID case counts (via date), and NOAA weather data (via date).
The join logic is handled in `pipeline.ipynb` Step 1.

---

## Expected `data/` folder after setup

```
data/
├── README_data.md               ← this file
├── lapd_crime.csv               
├── acs_blockgroup.csv           
├── la_covid_cases.csv           
├── noaa_weather.csv             
└── crime_with_acs_and_more.csv  ← final merged file (produced by pipeline step 1)
```