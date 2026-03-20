# -*- coding: utf-8 -*-
"""
Prediction App page — loaded by app.py via st.navigation.
Supports multiple model backends via model_registry.py.
Precipitation can be auto-fetched from Open-Meteo (weather.py).
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import sys
import os

# Ensure the project root is on the path so local modules are importable
_PAGES_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PAGES_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from features import VALID_LOC_NUMBERS, MAX_PRECIPITATION, FEATURES, LOC_LABELS
from model_registry import MODEL_REGISTRY, MODEL_DISPLAY_ORDER
from location_coords import get_coords, LOCATION_COORDS
from weather import get_precipitation

# ---------------------------------------------------------------------------
# Model loading (cached per model_key)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model(model_key: str):
    """Load and cache a model by its registry key. Called once per key."""
    wrapper = MODEL_REGISTRY[model_key]
    return wrapper.load_fn(wrapper.model_file)


# Preload all registered models at startup
loaded_models: dict = {}
for _key in MODEL_DISPLAY_ORDER:
    _wrapper = MODEL_REGISTRY[_key]
    if not _wrapper.exists:
        st.error(
            f"**Model artifact not found:** `{_wrapper.model_file}`\n\n"
            f"Run `python train_models.py` to generate all model files, "
            f"then restart the app."
        )
        st.stop()
    try:
        loaded_models[_key] = load_model(_key)
    except Exception as e:
        st.error(
            f"**Could not load '{_wrapper.display_name}'**\n\n"
            f"File: `{_wrapper.model_file}`\n\nError: {e}"
        )
        st.stop()

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_raw_row(
    loc_number: int,
    input_date: datetime.date,
    precipitation: float,
) -> pd.DataFrame:
    """Build a one-row raw DataFrame from manual entry inputs."""
    return pd.DataFrame({
        "Loc Number":    [loc_number],
        "Date":          [pd.Timestamp(input_date)],
        "Precipitation": [precipitation],
    })


def parse_uploaded_file(uploaded_file, require_precipitation: bool = True):
    """
    Read a CSV or Excel upload and validate it.
    Returns (df, errors) where errors is a list of strings.
    """
    errors = []
    name = uploaded_file.name.lower()

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            return None, ["Unsupported file type. Please upload a CSV or Excel file."]
    except Exception as e:
        return None, [f"Could not read file: {e}"]

    if len(df) == 0:
        return None, ["The uploaded file contains no data rows."]

    required = {"Loc Number", "Date"} if not require_precipitation else {"Loc Number", "Date", "Precipitation"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Missing required column(s): {', '.join(sorted(missing))}")
        return None, errors

    if "Precipitation" not in df.columns:
        df["Precipitation"] = float("nan")

    try:
        df["Loc Number"] = df["Loc Number"].astype(float).astype(int)
    except (ValueError, TypeError):
        errors.append(
            "'Loc Number' column contains non-numeric values. "
            "It must contain integers (e.g. 42)."
        )

    if require_precipitation:
        try:
            df["Precipitation"] = pd.to_numeric(df["Precipitation"], errors="raise")
        except (ValueError, TypeError):
            errors.append(
                "'Precipitation' column contains non-numeric values. "
                "It must contain numbers (e.g. 0.25)."
            )
    else:
        df["Precipitation"] = pd.to_numeric(df["Precipitation"], errors="coerce")

    parsed_dates = pd.to_datetime(df["Date"], errors="coerce")
    bad = parsed_dates.isna().sum()
    if bad > 0:
        errors.append(
            f"'Date' column has {bad} row(s) that could not be parsed. "
            "Use format YYYY-MM-DD (e.g. 2024-03-15)."
        )
    else:
        df["Date"] = parsed_dates

    if not any("Loc Number" in e for e in errors):
        invalid_locs = set(df["Loc Number"].unique()) - set(VALID_LOC_NUMBERS)
        if invalid_locs:
            errors.append(
                f"Loc Number value(s) not seen during training: {sorted(invalid_locs)}."
            )

    if not any("Precipitation" in e for e in errors):
        filled = df["Precipitation"].dropna() if not require_precipitation else df["Precipitation"]
        neg = (filled < 0).sum()
        if neg > 0:
            errors.append(
                f"'Precipitation' column has {neg} negative value(s). Must be >= 0."
            )

    if errors:
        return None, errors

    return df, []


def format_results(raw_df: pd.DataFrame, predictions: np.ndarray, pred_col: str = "Predicted GL Rev") -> pd.DataFrame:
    """Combine original input columns with predictions."""
    result = raw_df[["Loc Number", "Date", "Precipitation"]].copy()
    result[pred_col] = predictions
    result["Date"] = pd.to_datetime(result["Date"]).dt.date
    return result


def _pred_col_name(wrapper) -> str:
    if wrapper.target == "Unique Purchased Cards":
        return "Predicted Purchased Cards"
    return "Predicted GL Rev"


def _format_pred_value(val: float, wrapper) -> str:
    if wrapper.target == "Unique Purchased Cards":
        return f"{val:,.0f}"
    return f"${val:,.2f}"


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _do_autofill(loc_number: int, input_date: datetime.date, prefill_key: str):
    coords = get_coords(loc_number)
    if coords is None:
        st.warning(
            f"No coordinates configured for Location **{loc_number}**. "
            "Edit `location_coords.py` to enable weather auto-fill.",
            icon="⚠️",
        )
        return False

    with st.spinner(f"Fetching precipitation for {coords['name']} on {input_date}..."):
        result = get_precipitation(coords["lat"], coords["lon"], input_date)

    if result is None:
        st.error(
            "Could not retrieve weather data from Open-Meteo. "
            "Check your network connection or try a different date.",
            icon="🌐",
        )
        return False

    st.session_state[prefill_key] = result
    st.rerun()
    return True


# ---------------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------------

st.title("Revenue & Cards Predictor")
st.markdown(
    "Predict **Unique Purchased Cards** for any location by providing the date and "
    "forecasted precipitation. Choose **Manual Entry** for a single prediction "
    "or **File Upload** for batch predictions."
)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Model")
    selected_key = st.radio(
        "Choose prediction model:",
        options=MODEL_DISPLAY_ORDER,
        format_func=lambda k: MODEL_REGISTRY[k].display_name,
        index=0,
    )

    active_wrapper = MODEL_REGISTRY[selected_key]
    active_model   = loaded_models[selected_key]

    if active_wrapper.description:
        with st.expander("ℹ️ About this model", expanded=False):
            st.markdown(active_wrapper.description)

    st.divider()

    st.header("Prediction Mode")
    mode = st.radio(
        "Choose input method:",
        options=["Manual Entry", "File Upload"],
        index=0,
    )

    st.divider()
    st.caption(f"Model file: `{active_wrapper.filename}`")
    st.caption(f"Locations available: {len(VALID_LOC_NUMBERS):,}")

# ---------------------------------------------------------------------------
# Manual Entry
# ---------------------------------------------------------------------------

if mode == "Manual Entry":
    st.subheader(f"Single Prediction — {active_wrapper.display_name}")

    loc_mode = st.radio(
        "Location selection method:",
        options=["📋 Dropdown", "🗺️ Map"],
        index=0,
        horizontal=True,
        key="me_loc_mode",
        label_visibility="collapsed",
    )

    _ME_PREFILL = "_me_precip_prefill"
    if _ME_PREFILL in st.session_state:
        st.session_state["me_precip"] = float(st.session_state.pop(_ME_PREFILL))

    if loc_mode == "🗺️ Map":
        import pydeck as pdk

        map_df = pd.DataFrame([
            {
                "loc_number": loc,
                "lat": LOCATION_COORDS[loc]["lat"],
                "lon": LOCATION_COORDS[loc]["lon"],
                "name": LOCATION_COORDS[loc]["name"],
            }
            for loc in VALID_LOC_NUMBERS
            if loc in LOCATION_COORDS
        ])

        st.caption(
            "Click a store pin on the map to select it, then fill in the date "
            "and precipitation below."
        )

        _cur_loc = st.session_state.get("me_map_loc", VALID_LOC_NUMBERS[0])
        map_df["color"] = map_df["loc_number"].apply(
            lambda n: [255, 100, 0, 220] if n == _cur_loc else [30, 144, 255, 200]
        )
        map_df["radius"] = map_df["loc_number"].apply(
            lambda n: 40000 if n == _cur_loc else 25000
        )

        layer = pdk.Layer(
            "ScatterplotLayer",
            id="store-locations",
            data=map_df,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius="radius",
            pickable=True,
            auto_highlight=True,
            highlight_color=[255, 255, 0, 255],
            radius_min_pixels=8,
            radius_max_pixels=30,
        )

        view = pdk.ViewState(
            latitude=38.5,
            longitude=-96.0,
            zoom=3.2,
            pitch=0,
        )

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip={"text": "{name}\nLoc {loc_number}"},
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        )

        map_state = st.pydeck_chart(
            deck,
            selection_mode="single-object",
            on_select="rerun",
            key="me_map",
            use_container_width=True,
            height=450,
        )

        selected_objects = (
            map_state.selection.get("objects", {}).get("store-locations", [])
            if map_state and map_state.selection
            else []
        )
        if selected_objects:
            clicked_loc = int(selected_objects[0]["loc_number"])
            st.session_state["me_map_loc"] = clicked_loc
            st.session_state["me_map_selectbox"] = clicked_loc

        _has_selection = "me_map_loc" in st.session_state
        loc_number = st.session_state.get("me_map_loc", VALID_LOC_NUMBERS[0])

        if _has_selection:
            st.success(
                f"📍 **Selected location:** {LOCATION_COORDS[loc_number]['name']} "
                f"(Loc {loc_number})",
                icon="✅",
            )
        else:
            st.info(
                "👆 Click a blue pin on the map to select a store location.",
                icon="📍",
            )

        loc_number = st.selectbox(
            "Or pick / confirm location:",
            options=VALID_LOC_NUMBERS,
            format_func=lambda n: LOC_LABELS.get(n, str(n)),
            key="me_map_selectbox",
            help="Updates automatically when you click a pin, or choose manually here.",
        )
        st.session_state["me_map_loc"] = loc_number

    else:
        col_loc, _spacer = st.columns([1, 2])
        with col_loc:
            loc_number = st.selectbox(
                "Location",
                options=VALID_LOC_NUMBERS,
                index=0,
                format_func=lambda n: LOC_LABELS.get(n, str(n)),
                key="me_loc",
                help="Select from the 153 known store locations.",
            )

    col2, col3 = st.columns(2)
    with col2:
        input_date = st.date_input(
            "Date",
            value=datetime.date.today(),
            key="me_date",
            help="Date for which to predict Purchased Cards.",
        )
    with col3:
        precipitation = st.number_input(
            "Precipitation (inches)",
            min_value=0.0,
            max_value=1000.0,
            step=0.01,
            format="%.2f",
            key="me_precip",
            help="Forecasted daily precipitation in inches.",
        )

    autofill_col, predict_col, _ = st.columns([2, 1, 2])
    with autofill_col:
        if st.button(
            "Auto-fill Precipitation from Weather",
            key="me_autofill_btn",
            help="Fetches forecasted or historical daily precipitation "
                 "for this location and date from Open-Meteo (free, no key needed).",
        ):
            _do_autofill(loc_number, input_date, _ME_PREFILL)

    with predict_col:
        predict_clicked = st.button("Predict", type="primary", key="me_predict_btn")

    _max_precip = active_wrapper.max_precipitation
    if precipitation > _max_precip:
        st.warning(
            f"Precipitation ({precipitation:.2f} in) exceeds the training maximum "
            f"({_max_precip:.2f} in). The prediction may be less reliable."
        )

    if predict_clicked:
        try:
            raw_df = build_raw_row(loc_number, input_date, precipitation)
            preds  = active_wrapper.predict(raw_df, active_model)
            val    = preds[0]

            st.divider()
            st.subheader("Prediction Result")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Location", LOC_LABELS.get(loc_number, str(loc_number)))
            with c2:
                st.metric("Date", input_date.strftime("%m-%d-%Y"))
            with c3:
                st.metric("Precipitation", f"{precipitation:.2f} in")
            with c4:
                st.metric(
                    f"{_pred_col_name(active_wrapper)} ({active_wrapper.display_name})",
                    _format_pred_value(val, active_wrapper),
                )

        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------------------
# File Upload
# ---------------------------------------------------------------------------

elif mode == "File Upload":
    st.subheader(f"Batch Prediction — {active_wrapper.display_name}")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown(
            """
**Required columns:**

| Column | Type | Example |
|---|---|---|
| `Loc Number` | Integer | `42` |
| `Date` | Date (YYYY-MM-DD) | `2024-03-15` |
| `Precipitation` | Float >= 0 *(optional — can be auto-filled)* | `0.25` |

Extra columns are ignored.
            """
        )
    with col_b:
        sample_df = pd.DataFrame({
            "Loc Number":    [10, 14, 56],
            "Date":          ["2025-06-01", "2025-06-01", "2025-06-02"],
            "Precipitation": [0.00, 0.15, 1.20],
        })
        st.info(
            "The **Precipitation** column is now optional. "
            "If missing or blank, you can auto-fill it from Open-Meteo after uploading.",
            icon="🌧️",
        )
        st.download_button(
            label="Download Sample Template",
            data=convert_df_to_csv(sample_df),
            file_name="inference_template.csv",
            mime="text/csv",
        )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload your inference file",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file is not None:
        raw_df, errors = parse_uploaded_file(uploaded_file, require_precipitation=False)

        if errors:
            for err in errors:
                st.error(err)
            st.stop()

        if "_upload_df" in st.session_state:
            raw_df = st.session_state.pop("_upload_df")

        missing_mask  = raw_df["Precipitation"].isna()
        missing_count = missing_mask.sum()

        if missing_count > 0:
            st.info(
                f"**{missing_count} row(s)** have missing Precipitation values. "
                "Click the button below to auto-fill them from Open-Meteo.",
                icon="🌧️",
            )
            if st.button(
                f"Auto-fill {missing_count} Missing Precipitation Value(s)",
                key="upload_autofill_btn",
            ):
                filled_count = 0
                unknown_locs = set()
                failed_rows  = []

                progress = st.progress(0, text="Fetching weather data...")
                missing_indices = raw_df[missing_mask].index.tolist()

                for i, idx in enumerate(missing_indices):
                    row    = raw_df.loc[idx]
                    loc    = int(row["Loc Number"])
                    date   = row["Date"].date() if hasattr(row["Date"], "date") else row["Date"]
                    coords = get_coords(loc)

                    if coords is None:
                        unknown_locs.add(loc)
                        failed_rows.append(idx)
                    else:
                        result = get_precipitation(coords["lat"], coords["lon"], date)
                        if result is not None:
                            raw_df.at[idx, "Precipitation"] = result
                            filled_count += 1
                        else:
                            failed_rows.append(idx)

                    progress.progress(
                        (i + 1) / len(missing_indices),
                        text=f"Fetched {i + 1} of {len(missing_indices)} rows...",
                    )

                progress.empty()

                if unknown_locs:
                    st.warning(
                        f"No coordinates configured for location(s): "
                        f"**{sorted(unknown_locs)}**. "
                        "Edit `location_coords.py` to enable auto-fill for these stores.",
                        icon="⚠️",
                    )
                if filled_count > 0:
                    st.success(f"Auto-filled precipitation for **{filled_count}** row(s).")
                if len(failed_rows) - len(unknown_locs) > 0:
                    st.warning(
                        f"{len(failed_rows) - len(unknown_locs)} row(s) could not be "
                        "fetched (network error or date out of range).",
                        icon="🌐",
                    )

                st.session_state["_upload_df"] = raw_df
                st.rerun()

        still_missing = raw_df["Precipitation"].isna().sum()
        if still_missing > 0:
            st.warning(
                f"{still_missing} row(s) still have missing Precipitation values "
                "and will be excluded from predictions.",
                icon="⚠️",
            )
            raw_df = raw_df.dropna(subset=["Precipitation"])

        if len(raw_df) == 0:
            st.error("No rows remain after removing rows with missing Precipitation.")
            st.stop()

        try:
            preds = active_wrapper.predict(raw_df, active_model)
        except ValueError as ve:
            st.error(str(ve))
            st.stop()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        pred_col = _pred_col_name(active_wrapper)
        results_df = format_results(raw_df, preds, pred_col=pred_col)

        st.success(
            f"Predicted **{pred_col}** for **{len(results_df):,} row(s)** "
            f"using **{active_wrapper.display_name}**."
        )

        st.subheader("Prediction Results")
        display_df = results_df.copy()
        display_df["Date"] = pd.to_datetime(display_df["Date"]).dt.strftime("%m-%d-%Y")
        display_df[pred_col] = display_df[pred_col].apply(
            lambda x: _format_pred_value(x, active_wrapper)
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.subheader(f"{pred_col} by Location")
        chart_df = (
            results_df.groupby("Loc Number")[pred_col]
            .mean()
            .reset_index()
            .sort_values(pred_col, ascending=False)
            .set_index("Loc Number")
        )
        if len(chart_df) > 1:
            st.bar_chart(chart_df, y=pred_col, use_container_width=True)
            if len(results_df) > len(chart_df):
                st.caption(f"Chart shows mean predicted {pred_col} per location.")
        else:
            st.info("Upload data for more than one location to see a comparison chart.")

        st.divider()
        st.download_button(
            label="Download Results as CSV",
            data=convert_df_to_csv(results_df),
            file_name="predictions.csv",
            mime="text/csv",
            type="primary",
        )
