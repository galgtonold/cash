# Large-Scale Jupyter Projects for Cash Testing

> 10 data-intensive projects designed to stress-test the Cash notebook caching framework
> with real-world workloads (100s MB to low GBs of data).

## Status Tracker

| # | Project | Data Size | Status |
|---|---------|-----------|--------|
| 1 | NYC Taxi Trip Analysis | ~1 GB (Parquet) | ✅ Complete |
| 2 | Wikipedia Pageview Trends | ~3.89 GB (gzip TSV) | ✅ Complete |
| 3 | GitHub Archive Event Mining | ~5.90 GB (JSON.gz) | ✅ Complete |
| 4 | US Census ACS Demographic Analysis | ~839 MB (ZIP), 3.33 GB (CSV) | ✅ Complete |
| 5 | NOAA Global Weather Station Analysis | ~536 MB (CSV.gz), ~3.7 GB (RAM) | ✅ Complete |
| 6 | Stack Overflow Developer Survey Deep-Dive | ~50 MB (ZIP), 427 MB (CSV), 654 MB (RAM) | ✅ Complete |
| 7 | IMDb Movie & Rating Analysis | ~200 MB (TSV.gz), 6,790 MB (TSV), ~11.4 GB (RAM) | ✅ Complete |
| 8 | NYC 311 Service Request Analysis | ~2.1 GB (CSV), 3,081 MB (RAM), 3.46M rows | ✅ Complete |
| 9 | Kaggle Yelp Reviews NLP & ML | ~500 MB (JSON) | 🔲 Not Started |
| 10 | US Flights On-Time Performance | ~600 MB (CSV.gz) | 🔲 Not Started |

---

## Project Details

### 1. NYC Taxi Trip Analysis
**Goal**: Analyze ~1 GB of NYC yellow taxi trip data (Jan 2024). Compute fare distributions, tip percentages by time-of-day, popular pickup/dropoff zones, and trip duration heatmaps.
**Data**: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page (Parquet format)
**Key Jupyter Features**: Large DataFrame operations, groupby aggregations, pivot tables, matplotlib/seaborn visualizations, merge with taxi zone lookup table.
**Cash Stress Points**: Large DataFrame caching/restore, upstream invalidation across multiple dependent cells, file dependency tracking on downloaded parquet.

### 2. Wikipedia Pageview Trends
**Goal**: Analyze Wikipedia pageview data to identify trending topics, seasonal patterns, and cross-language popularity. Build time series decomposition of top pages.
**Data**: https://dumps.wikimedia.org/other/pageview_complete/ (~500 MB compressed TSVs)
**Key Jupyter Features**: Chunked CSV reading, string processing, time series analysis, rolling windows, multi-panel plots.
**Cash Stress Points**: Chunked I/O caching, large string DataFrames, multiple file dependencies.

### 3. GitHub Archive Event Mining
**Goal**: Mine GH Archive data to analyze repository activity patterns — stars, forks, PRs, issues by language/time. Build developer activity profiles and project health scores.
**Data**: https://www.gharchive.org/ (hourly JSON.gz dumps, ~30 MB each)
**Key Jupyter Features**: JSON parsing, nested data flattening, complex groupby operations, network-style analysis.
**Cash Stress Points**: Multiple file ingestion, JSON parsing overhead, iterative exploratory analysis.

### 4. US Census ACS Demographic Analysis
**Goal**: Analyze American Community Survey microdata to study income inequality, education-employment correlations, housing cost burden, and geographic mobility patterns across US states.
**Data**: https://www.census.gov/programs-surveys/acs/microdata.html (PUMS CSV files)
**Key Jupyter Features**: Survey weighting, cross-tabulations, geographic aggregations, statistical testing, choropleth maps.
**Cash Stress Points**: Multiple large CSV files, weighted aggregations, conditional analysis branches.

### 5. NOAA Global Weather Station Analysis
**Goal**: Process global weather station data to identify climate trends, extreme weather events, and station-level anomaly detection. Compute 30-year normals vs recent years.
**Data**: https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily (GHCN-Daily)
**Key Jupyter Features**: Time series processing, missing data interpolation, spatial joins, anomaly detection algorithms.
**Cash Stress Points**: Very wide date ranges, station-level groupby on millions of rows, iterative parameter tuning.

### 6. Stack Overflow Developer Survey Deep-Dive
**Goal**: Analyze the annual Stack Overflow developer survey to model salary prediction, technology adoption trends, and developer satisfaction factors using ML pipelines.
**Data**: https://survey.stackoverflow.co/ (annual survey CSV, ~200 MB)
**Key Jupyter Features**: Feature engineering, one-hot encoding, scikit-learn pipelines, cross-validation, SHAP explanations.
**Cash Stress Points**: ML pipeline caching (model objects), feature engineering chains, hyperparameter search loops.

### 7. IMDb Movie & Rating Analysis
**Goal**: Analyze the full IMDb non-commercial datasets — titles, ratings, names, crew, principals. Build genre popularity trends, rating prediction models, and actor/director collaboration networks using matrix operations.
**Data**: https://datasets.imdbws.com/ (TSV.gz files: title.basics, title.ratings, name.basics, title.principals, title.crew)
**Key Jupyter Features**: Multi-file joins, string parsing (genres/professions are comma-delimited), sparse matrix operations, ML rating prediction, large groupby aggregations.
**Cash Stress Points**: Multiple large DataFrame joins, string column operations, iterative merges, model training on sparse features.

### 8. NYC 311 Service Request Analysis
**Goal**: Analyze millions of NYC 311 service requests to identify complaint patterns by borough/time/type, response time analysis, seasonal trends, and predictive modeling of resolution time.
**Data**: https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9 (bulk CSV download)
**Key Jupyter Features**: Large CSV parsing, datetime operations, geographic groupby, time series decomposition, classification models.
**Cash Stress Points**: Very large single-file loading (millions of rows), datetime parsing overhead, complex multi-column groupby chains.

### 9. Kaggle Yelp Reviews NLP & ML
**Goal**: Analyze Yelp business and review data — build sentiment classifiers, topic modeling with TF-IDF + SVD, and business recommendation scoring using collaborative filtering patterns.
**Data**: Generated synthetic Yelp-like review dataset (~500 MB, mimicking Yelp Open Dataset structure)
**Key Jupyter Features**: Text processing (TF-IDF), dimensionality reduction (TruncatedSVD), sparse matrices, ML classification pipelines.
**Cash Stress Points**: Large text vectorization matrices, sparse matrix caching, iterative model tuning, pipeline object serialization.

### 10. US Flights On-Time Performance
**Goal**: Analyze Bureau of Transportation Statistics on-time flight data. Build delay prediction models, route analysis, carrier comparison, and seasonal pattern detection across millions of flights.
**Data**: https://www.transtats.bts.gov/DL_SelectFields.aspx (on-time performance CSV, or generated synthetic equivalent)
**Key Jupyter Features**: Large CSV processing, carrier/airport code joins, time series by route, ML delay classification, feature engineering.
**Cash Stress Points**: Multi-month data concatenation, large feature engineering chains, classification model caching, cross-validation loops.

---

## Testing Protocol

For each project:
1. **Setup**: Download data, verify file sizes, configure Cash
2. **Development**: Build notebook interactively, cell by cell
3. **Deliberate Bugs**: Introduce 2-3 intentional errors to trigger re-execution
4. **Cache Validation**: Verify correct cache behavior (COMPUTED/RESTORED/SKIPPED)
5. **Performance Logging**: Note cache overhead vs computation time ratios
6. **Bug Documentation**: Record any Cash bugs or unexpected behavior

## Findings Log

> Issues and observations will be documented here as projects are completed.

*(To be filled during project execution)*
