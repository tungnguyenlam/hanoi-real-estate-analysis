# Hanoi Real Estate Analysis

## Overview

This project analyzes the real estate market in Hanoi using data scraped from [Guland.vn](https://guland.vn). The dataset contains over 24,000 property listings, which were cleaned, processed, and analyzed to uncover insights about pricing trends, geographic distribution, and urban planning influences.

## Folder Structure

```
hanoi-real-estate-analysis/
│
├── data/                     # Contains raw and processed datasets
|---|-- figures/              # Contains figures
│   ├── guland_hanoi_listings_arcgis1.csv  # Raw dataset
│   ├── guland_hanoi_listings_arcgis2.csv  # Cleaned dataset
│
├── notebook/                 # Jupyter Notebooks for data processing and analysis
│   ├── 1_webscrape.ipynb            # Data scraping and initial data cleaning
│   ├── s1_data_cleaning.ipynb            # Data cleaning and preprocessing
│   ├── s2_exploratory_analysis.ipynb     # Exploratory data analysis
|   |-- s3_generate_features_prepare.ipynb
|   |-- s4_generate_features.ipynb
│
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

## Environment Setup

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/tungnguyenlam/hanoi-real-estate-analysis.git
   cd hanoi-real-estate-analysis
   ```

2. **Set Up a Virtual Environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install the dataset**

- Make sure you are in hanoi-real-esatate-analysis/

  ```bash
  chmod +x ./set_up_dataset.sh
  ./set_up_dataset.sh
  ```

## Key Features

### notebook/s1_data_cleaning.ipynb

- **Data Cleaning**: Handles missing values, duplicates, and ambiguous price formats.
- **Outlier Removal**: Uses the IQR method to filter extreme values.

### notebook/s2_data_visualization.ipynb

- **Geospatial Analysis**: Calculates distances from Ba Dinh Square to analyze geographic trends.
- **Exploratory Analysis**: Visualizes price distributions, correlations, and geographic heatmaps.

## Future Work

- Expand the dataset with additional sources.
- Integrate infrastructure and urban planning data.
- Develop predictive models for housing prices.

For more details, refer to the Jupyter Notebooks in the `notebook/` folder.

```

```
