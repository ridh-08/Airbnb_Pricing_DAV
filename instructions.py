"""
You are an expert data scientist. Build a complete, clean, modular data analytics pipeline in Python for an Airbnb pricing analysis project comparing New York and Chicago using InsideAirbnb datasets.

PROJECT GOAL:
Analyze Airbnb pricing dynamics, identify key determinants, perform clustering-based segmentation, and generate spatial and visual insights.

DATA:
Two cities: New York and Chicago

Each city has:
- listings.csv.gz (main dataset)
- reviews.csv.gz (for demand proxy)
- calendar.csv.gz (optional, large)
- neighbourhoods.geojson (for mapping)

DATA PATHS:
data/raw/newyork/
data/raw/chicago/

OUTPUT PATHS:
data/processed/
outputs/plots/
outputs/tables/

-----------------------------------
STEP 1: DATA LOADING
-----------------------------------
- Load compressed .csv.gz files using pandas
- Load geojson using geopandas
- Create reusable functions for loading city data

-----------------------------------
STEP 2: DATA CLEANING
-----------------------------------
- Convert price column (remove '$' and commas → float)
- Drop or impute missing values appropriately
- Remove extreme outliers in price (e.g., beyond 99th percentile)
- Ensure consistent column names across both datasets
- Keep only relevant columns:
    price, room_type, neighbourhood, latitude, longitude,
    accommodates, number_of_reviews, availability_365, amenities

-----------------------------------
STEP 3: FEATURE ENGINEERING
-----------------------------------
Create new features:
- log_price = log(price)
- price_per_person = price / accommodates
- amenities_count = count number of amenities from string
- review_density = number_of_reviews / availability_365 (handle division safely)

-----------------------------------
STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
-----------------------------------
Generate and save plots:
- Price distribution histogram (NY vs Chicago)
- Boxplot of price by room_type
- Correlation heatmap (numerical features)
- Scatter plots:
    price vs availability_365
    price vs number_of_reviews

Save plots to outputs/plots/

-----------------------------------
STEP 5: CLUSTERING (K-MEANS)
-----------------------------------
- Select features:
    log_price, availability_365, number_of_reviews, amenities_count
- Standardize features using StandardScaler
- Run KMeans for k = 3, 4, 5
- Evaluate using silhouette score
- Choose best k and assign cluster labels
- Create cluster summary table:
    mean price, availability, reviews, amenities per cluster

Save:
- clustered datasets
- cluster summary table

-----------------------------------
STEP 6: SPATIAL ANALYSIS
-----------------------------------
- Aggregate average price per neighbourhood
- Merge with geojson data
- Create:
    - Choropleth map (price by neighbourhood)
    - Cluster-based map (color-coded)
- Use Plotly or Folium

-----------------------------------
STEP 7: (OPTIONAL) CALENDAR ANALYSIS
-----------------------------------
- Sample a subset of calendar.csv.gz (due to size)
- Convert price and date
- Analyze:
    - availability trends

-----------------------------------
STEP 8: OUTPUTS
-----------------------------------
Save:
- cleaned datasets → data/processed/
- plots → outputs/plots/
- tables → outputs/tables/

-----------------------------------
STEP 9: CODE STRUCTURE
-----------------------------------
- Write modular, reusable functions
- Use clear variable names
- Add comments explaining each step
- Ensure code runs end-to-end

-----------------------------------
STEP 10: INSIGHTS (IMPORTANT)
-----------------------------------
For each analysis, print short insights like:
- "NY shows higher price dispersion than Chicago"
- "Cluster 2 represents high-price, low-availability listings"

-----------------------------------

Ensure the code is:
- clean and readable
- logically structured
- efficient (handle large files carefully)
- ready for presentation/demo

Start implementing step by step.
"""