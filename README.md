# Revenue & Cards Predictor

A Streamlit web app that predicts **Unique Purchased Cards** for store locations using a LightGBM model, and provides an interactive historical data analysis dashboard.

## Features

- **Prediction App** — Manual entry (with map or dropdown location selector) or batch file upload. Auto-fills precipitation from Open-Meteo weather API.
- **Data Analysis** — Interactive time-series charts across 153 store locations. Supports Daily / Weekly / Monthly / Quarterly intervals, plus a Special Dates mode covering 18 event types (holidays, Super Bowl, Black Friday, etc.) across multiple years.

## Project Structure

```
├── app.py                        # Streamlit entry point (navigation launcher)
├── pages/
│   ├── prediction_app.py         # Prediction page
│   └── analysis.py               # Historical data analysis page
├── features.py                   # Feature constants & location labels
├── model_registry.py             # Model wrapper & registry
├── location_coords.py            # Store lat/lon coordinates
├── weather.py                    # Open-Meteo precipitation API client
├── train_models.py               # Train all models (run once before first launch)
├── lgbm_model.py                 # Standalone LightGBM GL Rev training script
├── generate_test_comparison.py   # Generate test-set comparison CSV
├── historical_records.py         # Historical data utilities
├── update_data_file.py           # Data update utility
├── lgbm_cards_model.txt          # Pre-trained LightGBM (Purchased Cards)
├── lgbm_gl_rev_model.txt         # Pre-trained LightGBM (GL Revenue)
├── linreg_cards_model.joblib     # Pre-trained Linear Regression (Purchased Cards)
├── linreg_gl_rev_model.joblib    # Pre-trained Linear Regression (GL Revenue)
├── 20260202 Walk in sales - v1400.xlsx  # Source data (WI sales + model data)
└── requirements.txt              # Python dependencies
```

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the app
The pre-trained model artifacts are included in the repo, so you can launch immediately:
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

### 5. (Optional) Retrain models
If you have updated data, retrain all models by running:
```bash
python train_models.py
```
Then restart the Streamlit app.

## Requirements

- Python 3.10+
- See `requirements.txt` for all package dependencies

## Model Performance (test set)

| Model | Target | R² | MAE | RMSE |
|---|---|---|---|---|
| LightGBM | Unique Purchased Cards | 0.794 | 67.5 | 100.1 |
| LightGBM | GL Revenue | — | — | — |
| Linear Regression | Unique Purchased Cards | 0.489 | 120.3 | 157.8 |
| Linear Regression | GL Revenue | — | — | — |

## Data

The app reads from **`20260202 Walk in sales - v1400.xlsx`** which contains:
- `WI sales` — Raw walk-in sales data used for historical analysis
- `Model Data` — Cleaned GL Revenue training data
- `Data Model Cards` — Cleaned Purchased Cards training data
- `Location Lookup` — Store location reference data
