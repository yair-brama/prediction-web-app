# -*- coding: utf-8 -*-
"""
Test Data Analysis page — loaded by app.py via st.navigation.
Visualises test-set prediction results for the LightGBM new-precipitation
model, including per-location Pct Error time series and cross-location
metric comparison.
"""

import os
import sys
import datetime
import shutil
import tempfile

# ---------------------------------------------------------------------------
# Add the project root (parent of pages/) to sys.path so local modules resolve.
# Must be done BEFORE any local imports.
# ---------------------------------------------------------------------------
_PAGES_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PAGES_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

from features import LOC_LABELS

# Allow Altair to handle datasets larger than 5 000 rows (daily data with
# multiple locations selected can easily exceed the default limit).
alt.data_transformers.disable_max_rows()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EXCEL_FILE = os.path.join(_PROJECT_ROOT, "v3_lag_model_test_results.xlsx")

_RESAMPLE_FREQ = {
    "Daily":     "D",
    "Weekly":    "W",
    "Monthly":   "ME",
    "Quarterly": "QE",
}

_COMPARISON_METRICS = ["MAE", "RMSE", "R2", "MAPE (%)"]


# ---------------------------------------------------------------------------
# Special-dates helper functions (duplicated from analysis.py)
# ---------------------------------------------------------------------------

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> datetime.date:
    """Return the nth occurrence (1-based) of weekday (0=Mon...6=Sun) in year/month."""
    first = datetime.date(year, month, 1)
    delta = (weekday - first.weekday()) % 7
    return first + datetime.timedelta(days=delta + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> datetime.date:
    """Return the last occurrence of weekday in year/month."""
    if month == 12:
        last = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
    else:
        last = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
    delta = (last.weekday() - weekday) % 7
    return last - datetime.timedelta(days=delta)


def _good_friday(year: int) -> datetime.date:
    """Compute Good Friday using the Anonymous Gregorian algorithm for Easter."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day   = ((h + l - 7 * m + 114) % 31) + 1
    easter = datetime.date(year, month, day)
    return easter - datetime.timedelta(days=2)


# Canonical event-type names used for grouping in the filter UI
_EVENT_TYPES = [
    "New Year's Day",
    "New Year's Eve",
    "Good Friday",
    "Valentine's Day",
    "Independence Day",
    "MLK Day",
    "Presidents Day",
    "Memorial Day",
    "Mother's Day",
    "Labor Day",
    "Columbus Day",
    "Veterans Day",
    "Thanksgiving",
    "Black Friday",
    "Christmas",
    "Super Bowl",
]


def build_special_dates(years: list) -> dict:
    """
    Returns an ordered dict: event_label (str) -> date (datetime.date).
    Each label has the format "<Event Type> <Year>" (e.g. "Good Friday 2024").
    """
    SUPER_BOWL = {
        2023: datetime.date(2023, 2, 12),
        2024: datetime.date(2024, 2, 11),
        2025: datetime.date(2025, 2,  9),
        2026: datetime.date(2026, 2,  8),
    }

    special: dict = {}
    for y in sorted(years):
        special[f"New Year's Day {y}"]   = datetime.date(y,  1,  1)
        special[f"Valentine's Day {y}"]  = datetime.date(y,  2, 14)
        special[f"Independence Day {y}"] = datetime.date(y,  7,  4)
        special[f"Veterans Day {y}"]     = datetime.date(y, 11, 11)
        special[f"Christmas {y}"]        = datetime.date(y, 12, 25)
        special[f"New Year's Eve {y}"]   = datetime.date(y, 12, 31)

        special[f"Good Friday {y}"]      = _good_friday(y)

        special[f"MLK Day {y}"]          = _nth_weekday(y,  1, 0, 3)
        special[f"Presidents Day {y}"]   = _nth_weekday(y,  2, 0, 3)
        special[f"Memorial Day {y}"]     = _last_weekday(y, 5, 0)
        special[f"Mother's Day {y}"]     = _nth_weekday(y,  5, 6, 2)
        special[f"Labor Day {y}"]        = _nth_weekday(y,  9, 0, 1)
        special[f"Columbus Day {y}"]     = _nth_weekday(y, 10, 0, 2)
        thanksgiving                      = _nth_weekday(y, 11, 3, 4)
        special[f"Thanksgiving {y}"]     = thanksgiving
        special[f"Black Friday {y}"]     = thanksgiving + datetime.timedelta(days=1)

        if y in SUPER_BOWL:
            special[f"Super Bowl {y}"]   = SUPER_BOWL[y]

    return special


def _event_type(label: str) -> str:
    """Extract the event-type prefix from a label like 'Good Friday 2024' -> 'Good Friday'."""
    for et in _EVENT_TYPES:
        if label.startswith(et):
            return et
    return label.rsplit(" ", 1)[0]


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_test_predictions() -> pd.DataFrame:
    """Load the 'Test Predictions' sheet from the test results Excel."""
    try:
        df = pd.read_excel(_EXCEL_FILE, sheet_name="Test Predictions")
    except PermissionError:
        tmp = os.path.join(tempfile.gettempdir(), "capstone_test_results_copy.xlsx")
        shutil.copy2(_EXCEL_FILE, tmp)
        df = pd.read_excel(tmp, sheet_name="Test Predictions")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Pct Error"])
    df["Loc Number"] = df["Loc Number"].astype(int)
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _load_location_summary() -> pd.DataFrame:
    """Load the 'Location Summary' sheet from the test results Excel."""
    try:
        df = pd.read_excel(_EXCEL_FILE, sheet_name="Location Summary")
    except PermissionError:
        tmp = os.path.join(tempfile.gettempdir(), "capstone_test_results_copy.xlsx")
        shutil.copy2(_EXCEL_FILE, tmp)
        df = pd.read_excel(tmp, sheet_name="Location Summary")
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_pct_error(
    df: pd.DataFrame,
    interval: str,
    selected_locs: list[int],
    special_dates_map: dict,
    selected_event_types: list[str],
) -> pd.DataFrame:
    """
    Aggregate Pct Error for selected locations at the given interval.
    Uses MEAN (not sum) because Pct Error is a rate.
    """
    sub = df[df["Loc Number"].isin(selected_locs)].copy()

    if interval == "Special Dates":
        return _agg_special_dates(sub, selected_locs, special_dates_map, selected_event_types)

    freq = _RESAMPLE_FREQ[interval]
    rows: list[pd.DataFrame] = []
    for loc_num in selected_locs:
        loc_df = sub[sub["Loc Number"] == loc_num].set_index("Date")
        loc_label = LOC_LABELS.get(loc_num, str(loc_num))
        if loc_df.empty:
            continue
        grp = (
            loc_df.resample(freq)["Pct Error"]
            .mean()
            .reset_index()
        )
        grp = grp.dropna(subset=["Pct Error"])
        grp["Location"] = loc_label
        rows.append(grp)

    if not rows:
        return pd.DataFrame(columns=["Date", "Location", "Pct Error"])
    return pd.concat(rows, ignore_index=True)


def _agg_special_dates(
    sub: pd.DataFrame,
    selected_locs: list[int],
    special_dates_map: dict,
    selected_event_types: list[str],
) -> pd.DataFrame:
    """Filter to special-event dates, compute mean Pct Error per event per location."""
    sub = sub.copy()
    sub["Date"] = sub["Date"].dt.normalize()

    filtered_map = {
        label: d
        for label, d in special_dates_map.items()
        if _event_type(label) in selected_event_types
    }

    if not filtered_map:
        return pd.DataFrame(columns=["Event", "Event Type", "Location", "Pct Error"])

    special_ts  = {label: pd.Timestamp(d) for label, d in filtered_map.items()}
    ts_to_label = {v: k for k, v in special_ts.items()}

    filtered = sub[sub["Date"].isin(set(special_ts.values()))].copy()
    filtered["Event"]      = filtered["Date"].map(ts_to_label)
    filtered["Event Type"] = filtered["Event"].apply(_event_type)

    result_rows: list[pd.DataFrame] = []
    for loc_num in selected_locs:
        loc_label    = LOC_LABELS.get(loc_num, str(loc_num))
        loc_filtered = filtered[filtered["Loc Number"] == loc_num]
        if loc_filtered.empty:
            continue
        grp = (
            loc_filtered.groupby(["Event", "Event Type"])["Pct Error"]
            .mean()
            .reset_index()
        )
        grp["Location"] = loc_label
        result_rows.append(grp)

    if not result_rows:
        return pd.DataFrame(columns=["Event", "Event Type", "Location", "Pct Error"])
    return pd.concat(result_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _make_pct_error_line_chart(plot_df: pd.DataFrame) -> alt.Chart:
    """Interactive time-series line chart of Pct Error — one line per location."""
    return (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y(
                "Pct Error:Q",
                title="Mean Pct Error (%)",
                axis=alt.Axis(format=".1f"),
            ),
            color=alt.Color("Location:N", legend=alt.Legend(title="Location")),
            tooltip=[
                alt.Tooltip("Date:T",      title="Date",            format="%Y-%m-%d"),
                alt.Tooltip("Location:N",   title="Location"),
                alt.Tooltip("Pct Error:Q",  title="Pct Error (%)",  format=".1f"),
            ],
        )
        .properties(height=450)
        .interactive()
    )


def _make_pct_error_special_dates_chart(plot_df: pd.DataFrame, color_by: str) -> alt.Chart:
    """Grouped bar chart of Pct Error for Special Dates mode."""
    color_field = f"{color_by}:N"
    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "Event:N",
                title="Event",
                sort=None,
                axis=alt.Axis(labelAngle=-45, labelLimit=200),
            ),
            xOffset=alt.XOffset(f"{color_by}:N"),
            y=alt.Y(
                "Pct Error:Q",
                title="Mean Pct Error (%)",
                axis=alt.Axis(format=".1f"),
            ),
            color=alt.Color(color_field, legend=alt.Legend(title=color_by)),
            tooltip=[
                alt.Tooltip("Event:N",      title="Event"),
                alt.Tooltip("Event Type:N", title="Event Type"),
                alt.Tooltip("Location:N",   title="Location"),
                alt.Tooltip("Pct Error:Q",  title="Pct Error (%)", format=".1f"),
            ],
        )
        .properties(height=450)
    )


def _make_comparison_bar_chart(summary_df: pd.DataFrame, metric: str) -> alt.Chart:
    """Horizontal bar chart of locations sorted by a selected metric."""
    plot_df = summary_df[summary_df["Location"] != "** OVERALL **"].copy()

    # R2: higher is better (sort descending). Error metrics: lower is better (sort ascending).
    ascending = metric != "R2"
    sort_order = "ascending" if ascending else "descending"

    # Color scheme: green tones for R2, warm tones for error metrics
    scheme = "redyellowgreen" if metric == "R2" else "yelloworangered"
    reverse = metric != "R2"

    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            y=alt.Y(
                "Location:N",
                sort=alt.EncodingSortField(field=metric, order=sort_order),
                title="Location",
            ),
            x=alt.X(f"{metric}:Q", title=metric),
            color=alt.Color(
                f"{metric}:Q",
                scale=alt.Scale(scheme=scheme, reverse=reverse),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Location:N",   title="Location"),
                alt.Tooltip("Rows:Q",       title="Test Rows",  format=","),
                alt.Tooltip("MAE:Q",        title="MAE",        format=".2f"),
                alt.Tooltip("RMSE:Q",       title="RMSE",       format=".2f"),
                alt.Tooltip("R2:Q",         title="R\u00b2",    format=".4f"),
                alt.Tooltip("MAPE (%):Q",   title="MAPE (%)",   format=".1f"),
            ],
        )
        .properties(height=max(len(plot_df) * 18, 400))
        .interactive()
    )


def _make_comparison_scatter(summary_df: pd.DataFrame, x_metric: str, y_metric: str) -> alt.Chart:
    """Scatter plot comparing two metrics across locations."""
    plot_df = summary_df[summary_df["Location"] != "** OVERALL **"].copy()

    return (
        alt.Chart(plot_df)
        .mark_circle(size=80)
        .encode(
            x=alt.X(f"{x_metric}:Q", title=x_metric),
            y=alt.Y(f"{y_metric}:Q", title=y_metric),
            color=alt.Color("Location:N", legend=None),
            tooltip=[
                alt.Tooltip("Location:N",    title="Location"),
                alt.Tooltip("Rows:Q",        title="Test Rows",  format=","),
                alt.Tooltip(f"{x_metric}:Q", title=x_metric,     format=".2f"),
                alt.Tooltip(f"{y_metric}:Q", title=y_metric,     format=".2f"),
                alt.Tooltip("Avg Actual:Q",  title="Avg Actual", format=".1f"),
            ],
        )
        .properties(height=450)
        .interactive()
    )


# ===========================================================================
# PAGE BODY
# ===========================================================================

st.title("Test Data Analysis")
st.markdown(
    "Explore the test-set predictions of the **V3 Lag Model** — LightGBM with "
    "holiday features, lag/rolling features, and isotonic calibration. Compare "
    "prediction error across locations and time periods."
)
st.divider()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with st.spinner("Loading test results..."):
    df_predictions = _load_test_predictions()
    df_summary     = _load_location_summary()

if df_predictions.empty:
    st.error("No data could be loaded from the test results Excel file.")
    st.stop()

# Build special dates map from the years present in the test data
_data_years = sorted(df_predictions["Date"].dt.year.unique().tolist())
_SPECIAL_DATES_MAP = build_special_dates(_data_years)

# ===========================================================================
# Section 1: Prediction Error Time Series
# ===========================================================================

st.header("1. Prediction Error Time Series")
st.markdown(
    "Select one or more locations to visualise how **percentage prediction "
    "error** varies over the test period. Aggregate by different time "
    "intervals or filter to special events."
)

# --- Location multiselect (default: top 3 by row count) ---
_available_locs = (
    df_predictions.groupby("Loc Number")["Pct Error"]
    .count()
    .sort_values(ascending=False)
    .index
    .tolist()
)
_default_locs = _available_locs[:3]

selected_locs = st.multiselect(
    "Select Locations",
    options=_available_locs,
    default=_default_locs,
    format_func=lambda n: LOC_LABELS.get(n, str(n)),
    key="tda_loc_select",
    help="Choose one or more store locations. Defaults to the top 3 by test-set row count.",
)

if not selected_locs:
    st.warning("Please select at least one location.")
    st.stop()

# --- Interval + date-range / event-type controls ---
col_interval, col_right = st.columns([1, 2])

with col_interval:
    interval = st.radio(
        "Time Interval",
        options=["Daily", "Weekly", "Monthly", "Quarterly", "Special Dates"],
        index=2,
        key="tda_interval",
        help=(
            "**Daily** \u2014 one point per day  \n"
            "**Weekly / Monthly / Quarterly** \u2014 mean Pct Error over the period  \n"
            "**Special Dates** \u2014 filter to holidays & key events"
        ),
    )

with col_right:
    if interval == "Special Dates":
        selected_event_types = st.multiselect(
            "Filter by Event Type",
            options=_EVENT_TYPES,
            default=_EVENT_TYPES,
            key="tda_event_types",
        )
        color_by = st.radio(
            "Color bars by",
            options=["Location", "Event Type"],
            index=0,
            horizontal=True,
            key="tda_color_by",
        )
        df_filtered = df_predictions
    else:
        selected_event_types = _EVENT_TYPES
        color_by = "Location"

        min_date = df_predictions["Date"].min().date()
        max_date = df_predictions["Date"].max().date()
        date_range = st.slider(
            "Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key="tda_date_range",
        )
        df_filtered = df_predictions[
            (df_predictions["Date"] >= pd.Timestamp(date_range[0]))
            & (df_predictions["Date"] <= pd.Timestamp(date_range[1]))
        ]

# --- Guard ---
if interval == "Special Dates" and not selected_event_types:
    st.warning("Please select at least one event type.")
    st.stop()

# --- Aggregate ---
plot_df = _aggregate_pct_error(
    df_filtered, interval, selected_locs,
    _SPECIAL_DATES_MAP, selected_event_types,
)

if plot_df.empty:
    st.info("No data found for the selected filters.")
else:
    # --- Render chart ---
    if interval == "Special Dates":
        event_order = sorted(
            plot_df["Event"].unique(),
            key=lambda e: _SPECIAL_DATES_MAP.get(e, datetime.date.min),
        )
        plot_df["Event"] = pd.Categorical(
            plot_df["Event"], categories=event_order, ordered=True,
        )
        plot_df = plot_df.sort_values("Event")
        chart1 = _make_pct_error_special_dates_chart(plot_df, color_by)
    else:
        chart1 = _make_pct_error_line_chart(plot_df)

    st.altair_chart(chart1, use_container_width=True)

st.divider()

# ===========================================================================
# Section 2: Location Comparison
# ===========================================================================

st.header("2. Location Comparison")
st.markdown(
    "Compare model performance metrics across all locations. Use the "
    "**bar chart** to rank locations or the **scatter plot** to explore "
    "relationships between metrics."
)

# --- Overall metrics row ---
overall_row = df_summary[df_summary["Location"] == "** OVERALL **"]
if not overall_row.empty:
    ov = overall_row.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Overall MAE", f"{ov['MAE']:.2f}")
    with m2:
        st.metric("Overall RMSE", f"{ov['RMSE']:.2f}")
    with m3:
        st.metric("Overall R\u00b2", f"{ov['R2']:.4f}")
    with m4:
        st.metric("Overall MAPE", f"{ov['MAPE (%)']:.1f}%")

# --- Two tabs: Bar Chart and Scatter Plot ---
tab_bar, tab_scatter = st.tabs(["\U0001f4ca Ranked Bar Chart", "\U0001f4c8 Scatter Plot"])

with tab_bar:
    bar_metric = st.selectbox(
        "Rank locations by:",
        options=_COMPARISON_METRICS,
        index=0,
        key="tda_bar_metric",
    )
    chart_bar = _make_comparison_bar_chart(df_summary, bar_metric)
    st.altair_chart(chart_bar, use_container_width=True)

with tab_scatter:
    col_x, col_y = st.columns(2)
    _scatter_options = _COMPARISON_METRICS + ["Avg Actual", "Avg Predicted", "Rows"]
    with col_x:
        x_metric = st.selectbox(
            "X-axis metric:",
            options=_scatter_options,
            index=0,
            key="tda_scatter_x",
        )
    with col_y:
        y_metric = st.selectbox(
            "Y-axis metric:",
            options=_scatter_options,
            index=2,   # default to R2
            key="tda_scatter_y",
        )
    chart_scatter = _make_comparison_scatter(df_summary, x_metric, y_metric)
    st.altair_chart(chart_scatter, use_container_width=True)

st.divider()

# --- Raw data table ---
with st.expander("View Location Summary Table"):
    st.dataframe(
        df_summary,
        use_container_width=True,
        column_config={
            "Rows":          st.column_config.NumberColumn(format="%d"),
            "Avg Actual":    st.column_config.NumberColumn(format="%.1f"),
            "Avg Predicted": st.column_config.NumberColumn(format="%.1f"),
            "MAE":           st.column_config.NumberColumn(format="%.2f"),
            "RMSE":          st.column_config.NumberColumn(format="%.2f"),
            "R2":            st.column_config.NumberColumn(format="%.4f"),
            "MAPE (%)":      st.column_config.NumberColumn(format="%.1f"),
        },
    )
