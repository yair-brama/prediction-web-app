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

    def predict(self, raw_df: pd.DataFrame, loaded_model: Any) -> np.ndarray:
        """
        End-to-end inference: apply feature engineering then predict.
        raw_df must have columns: Loc Number (int), Date (datetime-like), Precipitation (float).
        """
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
}

# Controls the order models appear in the sidebar radio.
# Only lgbm_cards is exposed in the UI; the others remain in the registry
# so their artifacts are still loadable but are not shown to users.
MODEL_DISPLAY_ORDER: list[str] = ["lgbm_cards"]
