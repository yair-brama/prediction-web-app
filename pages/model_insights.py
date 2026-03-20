# -*- coding: utf-8 -*-
"""
Model Insights page — loaded by app.py via st.navigation.
Provides LightGBM interpretability via feature importance and SHAP analysis,
with PDF export of all visualisations.

Supports ALL models in MODEL_DISPLAY_ORDER via a sidebar selector.
"""

import os
import sys
import io
import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Path setup (same pattern as prediction_app.py / analysis.py)
# ---------------------------------------------------------------------------

_PAGES_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PAGES_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from features import (
    FEATURES, FEATURES_V2, FEATURES_V3,
    VALID_LOC_NUMBERS, LOC_LABELS,
    TARGET_CARDS,
    engineer_features_lgbm, engineer_features_lgbm_v2,
    engineer_features_lgbm_v3,
)
from model_registry import MODEL_REGISTRY, MODEL_DISPLAY_ORDER

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SHAP_SAMPLE_SIZE = 2000
_RANDOM_SEED      = 42

# Data sources
_EXCEL_DATA_FILE = os.path.join(_PROJECT_ROOT, "20260202 Walk in sales - v1400.xlsx")
_EXCEL_SHEET     = "Data Model Cards"
_CSV_DATA_FILE   = os.path.join(_PROJECT_ROOT, "merged_data_model.csv")

# Map model keys to their feature lists
_MODEL_FEATURES = {
    "lgbm_cards":            FEATURES,
    "lgbm_cards_new_precip": FEATURES,
    "lgbm_cards_calibrated": FEATURES_V2,
    "lgbm_cards_lag":        FEATURES_V3,
}

# Map model keys to their engineer functions
_MODEL_ENGINEER_FN = {
    "lgbm_cards":            engineer_features_lgbm,
    "lgbm_cards_new_precip": engineer_features_lgbm,
    "lgbm_cards_calibrated": engineer_features_lgbm_v2,
    "lgbm_cards_lag":        engineer_features_lgbm_v3,
}


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_booster(model_key: str):
    """
    Load the LightGBM Booster for a given model key.
    Handles models that return tuples (booster, calibrator, ...) by extracting
    the raw booster.
    """
    import lightgbm as lgb

    wrapper = MODEL_REGISTRY[model_key]
    model_file = wrapper.model_file

    # Always load just the raw booster for SHAP/importance analysis
    return lgb.Booster(model_file=model_file)


@st.cache_data(show_spinner=False)
def _load_training_data(model_key: str):
    """
    Load training data and apply the correct feature engineering for the model.
    Returns (feature_df, target_array).
    """
    import shutil
    import tempfile

    if model_key == "lgbm_cards":
        # Original model uses Excel data source
        try:
            raw = pd.read_excel(_EXCEL_DATA_FILE, sheet_name=_EXCEL_SHEET)
        except PermissionError:
            tmp = os.path.join(tempfile.gettempdir(), "capstone_data_copy.xlsx")
            shutil.copy2(_EXCEL_DATA_FILE, tmp)
            raw = pd.read_excel(tmp, sheet_name=_EXCEL_SHEET)

        raw = raw.dropna(subset=[TARGET_CARDS])
        engineer_fn = _MODEL_ENGINEER_FN[model_key]
        feature_df = engineer_fn(raw)
        target = raw[TARGET_CARDS].values
        return feature_df, target

    else:
        # All newer models use the CSV data source
        raw = pd.read_csv(_CSV_DATA_FILE)
        raw = raw.rename(columns={
            "date":              "Date",
            "loc_number":        "Loc Number",
            "new_precipitation": "Precipitation",
        })

        if model_key == "lgbm_cards_lag":
            # V3 lag model: engineer_fn needs target column named
            # "Unique Purchased Cards" for lag computation
            raw = raw.rename(columns={"purchased_cards": "Unique Purchased Cards"})
            raw = raw.dropna(subset=["Unique Purchased Cards"])
            engineer_fn = _MODEL_ENGINEER_FN[model_key]
            # Pass model_bundle=None for training path (lags from data itself)
            feature_df = engineer_fn(raw, model_bundle=None)
            target = raw["Unique Purchased Cards"].values
        else:
            # V1 new-precip and V2 calibrated
            raw = raw.rename(columns={"purchased_cards": "Target"})
            raw = raw.dropna(subset=["Target"])
            engineer_fn = _MODEL_ENGINEER_FN[model_key]
            feature_df = engineer_fn(raw)
            target = raw["Target"].values

        return feature_df, target


@st.cache_data(show_spinner=False)
def _compute_shap_values(model_key: str, feature_df: pd.DataFrame):
    """
    Compute SHAP values on a sampled subset using TreeExplainer.
    Returns (shap_values, sample_df, expected_value).
    """
    import shap
    import lightgbm as lgb

    wrapper = MODEL_REGISTRY[model_key]
    booster = lgb.Booster(model_file=wrapper.model_file)
    explainer = shap.TreeExplainer(booster)

    if len(feature_df) > _SHAP_SAMPLE_SIZE:
        sample_df = feature_df.sample(n=_SHAP_SAMPLE_SIZE, random_state=_RANDOM_SEED)
    else:
        sample_df = feature_df.copy()

    shap_values = explainer.shap_values(sample_df)
    ev = explainer.expected_value
    expected_value = float(np.asarray(ev).flat[0])

    return shap_values, sample_df.reset_index(drop=True), expected_value


# ---------------------------------------------------------------------------
# Plot functions (each returns a matplotlib Figure)
# ---------------------------------------------------------------------------

def _plot_feature_importance_gain(booster) -> plt.Figure:
    """Horizontal bar chart of feature importance by gain."""
    names = booster.feature_name()
    gains = booster.feature_importance(importance_type="gain")
    order = np.argsort(gains)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(names)[order], gains[order], color="#4C78A8")
    ax.set_xlabel("Importance (Gain)")
    ax.set_title("Feature Importance by Gain")
    fig.tight_layout()
    return fig


def _plot_feature_importance_split(booster) -> plt.Figure:
    """Horizontal bar chart of feature importance by split count."""
    names = booster.feature_name()
    splits = booster.feature_importance(importance_type="split")
    order = np.argsort(splits)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(names)[order], splits[order], color="#F58518")
    ax.set_xlabel("Importance (Split Count)")
    ax.set_title("Feature Importance by Split Count")
    fig.tight_layout()
    return fig


def _plot_shap_summary(shap_values, sample_df) -> plt.Figure:
    """SHAP beeswarm / summary plot."""
    import shap

    plt.figure()
    shap.summary_plot(shap_values, sample_df, show=False, plot_size=None)
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    fig.tight_layout()
    return fig


def _plot_shap_bar(shap_values, sample_df) -> plt.Figure:
    """SHAP mean-absolute bar plot."""
    import shap

    plt.figure()
    shap.summary_plot(shap_values, sample_df, plot_type="bar", show=False, plot_size=None)
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    fig.tight_layout()
    return fig


def _plot_shap_dependence(shap_values, sample_df, feature_name: str) -> plt.Figure:
    """SHAP dependence plot for a selected feature."""
    import shap

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(feature_name, shap_values, sample_df, ax=ax, show=False)
    ax.set_title(f"SHAP Dependence: {feature_name}")
    fig.tight_layout()
    return fig


def _plot_shap_waterfall(shap_values, sample_df, expected_value, row_idx: int) -> plt.Figure:
    """SHAP waterfall plot for a single prediction."""
    import shap

    explanation = shap.Explanation(
        values=shap_values[row_idx],
        base_values=expected_value,
        data=sample_df.iloc[row_idx].values,
        feature_names=list(sample_df.columns),
    )
    plt.figure()
    shap.plots.waterfall(explanation, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def _generate_pdf(model_name: str, figures: list) -> bytes:
    """Generate a multi-page PDF from a list of (title, Figure) pairs."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Title page
        title_fig, title_ax = plt.subplots(figsize=(11, 8.5))
        title_ax.axis("off")
        title_ax.text(
            0.5, 0.6,
            "Model Insights Report",
            transform=title_ax.transAxes,
            fontsize=28, fontweight="bold", ha="center", va="center",
        )
        title_ax.text(
            0.5, 0.45,
            f"{model_name}\n"
            f"Unique Purchased Cards\n"
            f"Generated {datetime.date.today().strftime('%B %d, %Y')}",
            transform=title_ax.transAxes,
            fontsize=14, ha="center", va="center", color="gray",
        )
        pdf.savefig(title_fig, bbox_inches="tight", dpi=150)
        plt.close(title_fig)

        for title, fig in figures:
            fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
            pdf.savefig(fig, bbox_inches="tight", dpi=150)

    buf.seek(0)
    return buf.getvalue()


# ===========================================================================
# PAGE BODY
# ===========================================================================

st.title("Model Insights")
st.markdown(
    "Understand **how** the LightGBM models predict Unique Purchased Cards. "
    "Select a model below to view its feature importance, SHAP-based explanations, "
    "and inspect individual predictions. All charts can be exported to PDF."
)
st.divider()

# ---------------------------------------------------------------------------
# Model selector
# ---------------------------------------------------------------------------

model_options = MODEL_DISPLAY_ORDER
model_labels = {k: MODEL_REGISTRY[k].display_name for k in model_options}

selected_model_key = st.selectbox(
    "Select a model to analyze:",
    options=model_options,
    format_func=lambda k: model_labels[k],
    index=len(model_options) - 1,  # default to the latest (V3 lag)
    key="insights_model_select",
)

wrapper = MODEL_REGISTRY[selected_model_key]
feature_list = _MODEL_FEATURES[selected_model_key]

# Show model description
st.info(f"**{wrapper.display_name}**\n\n{wrapper.description}", icon="ℹ️")

# --- Guard: model artifact must exist ---
if not wrapper.exists:
    st.error(
        f"**Model artifact not found:** `{wrapper.model_file}`\n\n"
        "Please train the model first, then restart the app."
    )
    st.stop()

# --- Load model and data ---
booster = _load_booster(selected_model_key)

with st.spinner("Loading training data (may take up to 30 s on first load)..."):
    feature_df, target = _load_training_data(selected_model_key)

st.success(
    f"Loaded **{len(feature_df):,}** training rows across "
    f"**{len(VALID_LOC_NUMBERS)}** locations for **{wrapper.display_name}**.",
    icon="✅",
)

# Collect all figures for PDF export
pdf_figures: list = []

# -----------------------------------------------------------------------
# Section 1: Feature Importance
# -----------------------------------------------------------------------

st.header("1. Feature Importance")
st.markdown(
    "How much each feature contributes to the model's predictions, "
    "measured by two complementary metrics."
)

col_gain, col_split = st.columns(2)

with col_gain:
    st.subheader("Importance by Gain")
    st.caption(
        "Total reduction in the loss function contributed by all splits "
        "on this feature. Higher = more predictive power."
    )
    fig_gain = _plot_feature_importance_gain(booster)
    st.pyplot(fig_gain, use_container_width=True)
    pdf_figures.append(("Feature Importance (Gain)", fig_gain))

with col_split:
    st.subheader("Importance by Split Count")
    st.caption(
        "Number of times the model chose to split on this feature. "
        "Higher = more frequently used."
    )
    fig_split = _plot_feature_importance_split(booster)
    st.pyplot(fig_split, use_container_width=True)
    pdf_figures.append(("Feature Importance (Split Count)", fig_split))

st.divider()

# -----------------------------------------------------------------------
# Section 2: SHAP Analysis
# -----------------------------------------------------------------------

st.header("2. SHAP Analysis")
st.markdown(
    "SHAP (SHapley Additive exPlanations) values show **how much each feature "
    "pushes a prediction** above or below the model's average output. "
    f"Computed on a random sample of **{_SHAP_SAMPLE_SIZE:,}** rows."
)

with st.spinner("Computing SHAP values (this may take a few seconds on first load)..."):
    shap_values, sample_df, expected_value = _compute_shap_values(
        selected_model_key, feature_df,
    )

st.subheader("2a. SHAP Summary (Beeswarm)")
st.caption(
    "Each dot is one prediction. Position on the X-axis shows the SHAP value "
    "(impact on prediction). Color represents the feature value (red = high, blue = low)."
)
fig_beeswarm = _plot_shap_summary(shap_values, sample_df)
st.pyplot(fig_beeswarm, use_container_width=True)
pdf_figures.append(("SHAP Summary (Beeswarm)", fig_beeswarm))

st.subheader("2b. Mean Absolute SHAP Value")
st.caption("Average impact of each feature on model output magnitude.")
fig_bar = _plot_shap_bar(shap_values, sample_df)
st.pyplot(fig_bar, use_container_width=True)
pdf_figures.append(("Mean Absolute SHAP Value", fig_bar))

st.divider()

# -----------------------------------------------------------------------
# Section 3: SHAP Dependence Plot
# -----------------------------------------------------------------------

st.header("3. SHAP Dependence")
st.markdown(
    "Explore how a single feature's value relates to its SHAP value "
    "(its impact on the prediction). The color shows the interaction feature "
    "automatically chosen by SHAP."
)

dep_feature = st.selectbox(
    "Select feature for dependence plot:",
    options=feature_list,
    index=0,  # default to Precipitation
    key="dep_feature_select",
)

fig_dep = _plot_shap_dependence(shap_values, sample_df, dep_feature)
st.pyplot(fig_dep, use_container_width=True)
pdf_figures.append((f"SHAP Dependence: {dep_feature}", fig_dep))

st.divider()

# -----------------------------------------------------------------------
# Section 4: Single-Prediction Waterfall
# -----------------------------------------------------------------------

st.header("4. Single-Prediction Explanation")
st.markdown(
    "Select a specific data point from the sample to see how each feature "
    "contributes to that prediction, starting from the model's baseline "
    f"(average prediction: **{expected_value:.1f}** cards)."
)

col_loc, col_row = st.columns([1, 1])

with col_loc:
    # Get unique locations present in the sample
    if hasattr(sample_df["Loc Number"], "cat"):
        actual_locs = sorted(sample_df["Loc Number"].dropna().unique())
    else:
        actual_locs = sorted(sample_df["Loc Number"].unique())

    selected_loc = st.selectbox(
        "Filter by Location:",
        options=actual_locs,
        format_func=lambda n: LOC_LABELS.get(int(n), str(n)),
        index=0,
        key="waterfall_loc",
    )

with col_row:
    loc_mask = sample_df["Loc Number"] == selected_loc
    n_available = int(loc_mask.sum())
    row_number = st.number_input(
        f"Sample row (1 to {n_available}):",
        min_value=1,
        max_value=max(n_available, 1),
        value=1,
        step=1,
        key="waterfall_row",
    )

if n_available == 0:
    st.warning("No samples available for this location in the SHAP sample.")
else:
    # Map the user's 1-based row number to the position in shap_values
    loc_indices = sample_df[loc_mask].index.tolist()
    position = loc_indices[row_number - 1]

    # Show feature values for this row
    st.markdown("**Feature values for this prediction:**")
    row_data = sample_df.iloc[position]
    display_row = row_data.to_frame().T
    st.dataframe(display_row, use_container_width=True, hide_index=True)

    predicted_value = expected_value + shap_values[position].sum()
    st.metric("Predicted Unique Purchased Cards", f"{predicted_value:.0f}")

    fig_waterfall = _plot_shap_waterfall(shap_values, sample_df, expected_value, position)
    st.pyplot(fig_waterfall, use_container_width=True)
    pdf_figures.append(("Single Prediction Waterfall", fig_waterfall))

st.divider()

# -----------------------------------------------------------------------
# Section 5: PDF Export
# -----------------------------------------------------------------------

st.header("5. Export")
st.markdown(
    "Download all the charts displayed above as a single multi-page PDF document."
)

if pdf_figures:
    pdf_bytes = _generate_pdf(wrapper.display_name, pdf_figures)

    st.download_button(
        label=f"Download {wrapper.display_name} Insights as PDF",
        data=pdf_bytes,
        file_name=f"model_insights_{selected_model_key}.pdf",
        mime="application/pdf",
        type="primary",
    )

    # Clean up matplotlib figures to free memory
    for _, fig in pdf_figures:
        plt.close(fig)
else:
    st.info("No charts to export yet.")
