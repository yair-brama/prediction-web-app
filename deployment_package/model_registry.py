"""
model_registry.py
-----------------
Defines ModelWrapper and the registry of all available models.

To add a new model:
  1. Write a load_fn and predict_fn below.
  2. Add a ModelWrapper entry to MODEL_REGISTRY.
  3. Add its key to MODEL_DISPLAY_ORDER.
  4. No changes to app.py are needed.
"""

from __future__ import annotations
import dataclasses
import os
from typing import Callable, Any
import numpy as np
import pandas as pd

from features import (
    engineer_features_lgbm,
    engineer_features_lgbm_v2,
    engineer_features_lgbm_v3,
    engineer_features_sklearn,
)

# ---------------------------------------------------------------------------
# Base directory for all model artifacts
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# ModelWrapper
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ModelWrapper:
    """
    Thin adapter that presents a uniform interface over different model backends.

    Attributes
    ----------
    display_name : str
        Human-readable label shown in the sidebar radio.
    model_key : str
        Short slug used as the st.cache_resource key.
    model_file : str
        Absolute path to the serialized model artifact.
    engineer_fn : Callable
        Feature engineering function for this model.
        Signature: (raw_df: pd.DataFrame) -> pd.DataFrame
    load_fn : Callable
        Loads the artifact from disk and returns the raw model object.
        Signature: (model_file: str) -> Any
    predict_fn : Callable
        Runs inference.
        Signature: (model: Any, feature_df: pd.DataFrame) -> np.ndarray
    description : str
        Short description shown in the UI.
    """
    display_name: str
    model_key: str
    model_file: str
    engineer_fn: Callable
    load_fn: Callable
    predict_fn: Callable
    description: str = ""
    target: str = "GL Rev"   # "GL Rev" or "Unique Purchased Cards"
    max_precipitation: float = 782.49  # max value seen during training
    engineer_needs_model: bool = False  # if True, predict() passes loaded_model to engineer_fn

    def predict(self, raw_df: pd.DataFrame, loaded_model: Any) -> np.ndarray:
        """
        End-to-end inference: apply feature engineering then predict.
        raw_df must have columns: Loc Number (int), Date (datetime-like), Precipitation (float).
        """
        if self.engineer_needs_model:
            feature_df = self.engineer_fn(raw_df, loaded_model)
        else:
            feature_df = self.engineer_fn(raw_df)
        return self.predict_fn(loaded_model, feature_df)

    @property
    def filename(self) -> str:
        """Just the filename portion of model_file (for sidebar display)."""
        return os.path.basename(self.model_file)

    @property
    def exists(self) -> bool:
        """True if the artifact file is present on disk."""
        return os.path.isfile(self.model_file)


# ---------------------------------------------------------------------------
# LightGBM backend
# ---------------------------------------------------------------------------

def _load_lgbm(model_file: str):
    import lightgbm as lgb
    return lgb.Booster(model_file=model_file)


def _predict_lgbm(model, feature_df: pd.DataFrame) -> np.ndarray:
    # num_iteration=-1 uses all trees (best_iteration is -1 in this saved model)
    return model.predict(feature_df, num_iteration=model.best_iteration)


# ---------------------------------------------------------------------------
# LightGBM + Isotonic calibration backend
# ---------------------------------------------------------------------------

def _load_lgbm_calibrated(model_file: str):
    """Load the LightGBM booster + the companion isotonic calibrator."""
    import lightgbm as lgb
    import joblib
    booster = lgb.Booster(model_file=model_file)
    # Calibrator sits alongside the model file with a matching name
    cal_file = model_file.replace("_model.txt", "_calibrator.joblib")
    calibrator = joblib.load(cal_file)
    return (booster, calibrator)


def _predict_lgbm_calibrated(model_pair, feature_df: pd.DataFrame) -> np.ndarray:
    """Run LightGBM inference then apply isotonic calibration."""
    booster, calibrator = model_pair
    raw_preds = booster.predict(feature_df, num_iteration=booster.best_iteration)
    calibrated = calibrator.predict(raw_preds)
    return np.maximum(calibrated, 0)  # ensure non-negative


# ---------------------------------------------------------------------------
# LightGBM + Isotonic calibration + Lag history backend (V3)
# ---------------------------------------------------------------------------

def _load_lgbm_calibrated_v3(model_file: str):
    """Load LightGBM booster + isotonic calibrator + lag history."""
    import lightgbm as lgb
    import joblib
    booster = lgb.Booster(model_file=model_file)
    cal_file = model_file.replace("_model.txt", "_calibrator.joblib")
    calibrator = joblib.load(cal_file)
    hist_file = model_file.replace("_model.txt", "_history.parquet")
    history = pd.read_parquet(hist_file)
    return (booster, calibrator, history)


def _predict_lgbm_calibrated_v3(model_triple, feature_df: pd.DataFrame) -> np.ndarray:
    """Run LightGBM inference then apply isotonic calibration (V3)."""
    booster, calibrator, _history = model_triple
    raw_preds = booster.predict(feature_df, num_iteration=booster.best_iteration)
    calibrated = calibrator.predict(raw_preds)
    return np.maximum(calibrated, 0)


# ---------------------------------------------------------------------------
# scikit-learn backend (Pipeline with OHE + LinearRegression)
# ---------------------------------------------------------------------------

def _load_sklearn(model_file: str):
    import joblib
    return joblib.load(model_file)


def _predict_sklearn(model, feature_df: pd.DataFrame) -> np.ndarray:
    return model.predict(feature_df).astype(float)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelWrapper] = {
    "lgbm": ModelWrapper(
        display_name="LightGBM",
        model_key="lgbm",
        model_file=os.path.join(BASE_DIR, "lgbm_gl_rev_model.txt"),
        engineer_fn=engineer_features_lgbm,
        load_fn=_load_lgbm,
        predict_fn=_predict_lgbm,
        description="Gradient-boosted decision trees. Handles non-linear "
                    "interactions and location-specific patterns. (R2 ~ 0.80)",
    ),
    "linear_regression": ModelWrapper(
        display_name="Linear Regression",
        model_key="linear_regression",
        model_file=os.path.join(BASE_DIR, "linreg_gl_rev_model.joblib"),
        engineer_fn=engineer_features_sklearn,
        load_fn=_load_sklearn,
        predict_fn=_predict_sklearn,
        description="Ordinary least squares with one-hot encoded location. "
                    "Fast and interpretable baseline model.",
    ),
    "lgbm_cards": ModelWrapper(
        display_name="LightGBM (Purchased Cards)",
        model_key="lgbm_cards",
        model_file=os.path.join(BASE_DIR, "lgbm_cards_model.txt"),
        engineer_fn=engineer_features_lgbm,
        load_fn=_load_lgbm,
        predict_fn=_predict_lgbm,
        description=(
            "**R² = 0.794 · MAE = 68 cards · RMSE = 100 cards**\n\n"
            "LightGBM is a gradient-boosted decision tree model that learns "
            "complex, non-linear relationships between location, calendar features, "
            "and precipitation to predict the number of unique purchased cards for "
            "a given store and date.\n\n"
            "**Training data:** Feb 2023 – Jun 2025 (≈128 k rows)  \n"
            "**Test period:** Jun 2025 – Jan 2026 (≈32 k rows)  \n"
            "**Features:** Location, Day of Week, Month, Day of Month, "
            "Week of Year, Weekend flag, Quarter, Precipitation"
        ),
        target="Unique Purchased Cards",
    ),
    "linreg_cards": ModelWrapper(
        display_name="Linear Regression (Purchased Cards)",
        model_key="linreg_cards",
        model_file=os.path.join(BASE_DIR, "linreg_cards_model.joblib"),
        engineer_fn=engineer_features_sklearn,
        load_fn=_load_sklearn,
        predict_fn=_predict_sklearn,
        description="Ordinary least squares predicting Unique Purchased Cards "
                    "with one-hot encoded location.",
        target="Unique Purchased Cards",
    ),
    "lgbm_cards_new_precip": ModelWrapper(
        display_name="LightGBM — New Precipitation (Purchased Cards)",
        model_key="lgbm_cards_new_precip",
        model_file=os.path.join(BASE_DIR, "lgbm_cards_new_precip_model.txt"),
        engineer_fn=engineer_features_lgbm,
        load_fn=_load_lgbm,
        predict_fn=_predict_lgbm,
        description=(
            "**R² = 0.799 · MAE = 66 cards · RMSE = 99 cards**\n\n"
            "Same LightGBM architecture as the original model, but trained on "
            "**accurate historical precipitation data** (in inches) sourced from "
            "Open-Meteo, replacing the original precipitation values.\n\n"
            "**Training data:** Feb 2023 – Jun 2025 (≈128 k rows)  \n"
            "**Test period:** Jun 2025 – Jan 2026 (≈32 k rows)  \n"
            "**Features:** Location, Day of Week, Month, Day of Month, "
            "Week of Year, Weekend flag, Quarter, Precipitation (inches)"
        ),
        target="Unique Purchased Cards",
        max_precipitation=7.71,
    ),
    "lgbm_cards_calibrated": ModelWrapper(
        display_name="LightGBM — Calibrated V2 (Purchased Cards)",
        model_key="lgbm_cards_calibrated",
        model_file=os.path.join(BASE_DIR, "lgbm_cards_calibrated_model.txt"),
        engineer_fn=engineer_features_lgbm_v2,
        load_fn=_load_lgbm_calibrated,
        predict_fn=_predict_lgbm_calibrated,
        description=(
            "Improved LightGBM model with two key enhancements:\n\n"
            "**1. Holiday features** — IsHoliday, IsHolidayWeekend, "
            "DaysToNearestHoliday (US federal holidays)\n\n"
            "**2. Bias calibration** — Isotonic regression post-processing "
            "to correct the systematic over-prediction bias\n\n"
            "**Training data:** Feb 2023 – Jun 2025 (70% train, 10% calibration)  \n"
            "**Test period:** Jun 2025 – Jan 2026 (20%)  \n"
            "**Features:** Location, Day of Week, Month, Day of Month, "
            "Week of Year, Weekend flag, Quarter, Precipitation (inches), "
            "IsHoliday, IsHolidayWeekend, DaysToNearestHoliday"
        ),
        target="Unique Purchased Cards",
        max_precipitation=7.71,
    ),
    "lgbm_cards_lag": ModelWrapper(
        display_name="LightGBM \u2014 Lag V3 (Purchased Cards)",
        model_key="lgbm_cards_lag",
        model_file=os.path.join(BASE_DIR, "lgbm_cards_lag_model.txt"),
        engineer_fn=engineer_features_lgbm_v3,
        load_fn=_load_lgbm_calibrated_v3,
        predict_fn=_predict_lgbm_calibrated_v3,
        description=(
            "Best short-term model with three layers of improvements:\n\n"
            "**1. Lag features** \u2014 Lag-7, Lag-14, 7-day and 14-day "
            "rolling mean per location (r=0.74 with target)\n\n"
            "**2. Holiday features** \u2014 IsHoliday, IsHolidayWeekend, "
            "DaysToNearestHoliday\n\n"
            "**3. Bias calibration** \u2014 Isotonic regression post-processing\n\n"
            "**Best for:** predicting tomorrow / this week when recent "
            "sales data is available.\n\n"
            "**Features:** All V2 features + Lag7, Lag14, Roll7Mean, Roll14Mean"
        ),
        target="Unique Purchased Cards",
        max_precipitation=7.71,
        engineer_needs_model=True,
    ),
}

# Controls the order models appear in the sidebar radio.
MODEL_DISPLAY_ORDER: list[str] = [
    "lgbm_cards_lag",
]
