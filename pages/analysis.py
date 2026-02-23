# -*- coding: utf-8 -*-
"""
Historical Data Analysis page — loaded by app.py via st.navigation.
Visualises Unique Purchased Cards from the "WI sales" Excel sheet across
multiple store locations with configurable time intervals and special-event
filtering.
"""

import os
import sys
import datetime

# ---------------------------------------------------------------------------
# Add the project root (parent of pages/) to sys.path so local modules resolve.
# Must be done BEFORE any local imports.
# ---------------------------------------------------------------------------
_PAGES_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PAGES_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import altair as alt
import streamlit as st

from features import VALID_LOC_NUMBERS, LOC_LABELS

# ---------------------------------------------------------------------------
# Special-dates helper functions
# ---------------------------------------------------------------------------

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> datetime.date:
    """Return the nth occurrence (1-based) of weekday (0=Mon…6=Sun) in year/month."""
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
    l = (32 + 2 * e + 2 * i - h - k) % 7
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
    The event-type prefix matches an entry in _EVENT_TYPES so the UI can
    group / filter by event type.
    """
    SUPER_BOWL = {
        2023: datetime.date(2023, 2, 12),
        2024: datetime.date(2024, 2, 11),
        2025: datetime.date(2025, 2,  9),
    }

    special: dict = {}
    for y in sorted(years):
        # Fixed-date holidays
        special[f"New Year's Day {y}"]   = datetime.date(y,  1,  1)
        special[f"Valentine's Day {y}"]  = datetime.date(y,  2, 14)
        special[f"Independence Day {y}"] = datetime.date(y,  7,  4)
        special[f"Veterans Day {y}"]     = datetime.date(y, 11, 11)
        special[f"Christmas {y}"]        = datetime.date(y, 12, 25)
        special[f"New Year's Eve {y}"]   = datetime.date(y, 12, 31)

        # Good Friday (2 days before Easter, calculated)
        special[f"Good Friday {y}"]      = _good_friday(y)

        # Calculated floating holidays
        special[f"MLK Day {y}"]          = _nth_weekday(y,  1, 0, 3)   # 3rd Mon Jan
        special[f"Presidents Day {y}"]   = _nth_weekday(y,  2, 0, 3)   # 3rd Mon Feb
        special[f"Memorial Day {y}"]     = _last_weekday(y, 5, 0)      # Last Mon May
        special[f"Mother's Day {y}"]     = _nth_weekday(y,  5, 6, 2)   # 2nd Sun May
        special[f"Labor Day {y}"]        = _nth_weekday(y,  9, 0, 1)   # 1st Mon Sep
        special[f"Columbus Day {y}"]     = _nth_weekday(y, 10, 0, 2)   # 2nd Mon Oct
        thanksgiving                      = _nth_weekday(y, 11, 3, 4)  # 4th Thu Nov
        special[f"Thanksgiving {y}"]     = thanksgiving
        special[f"Black Friday {y}"]     = thanksgiving + datetime.timedelta(days=1)

        # Super Bowl (hardcoded per year)
        if y in SUPER_BOWL:
            special[f"Super Bowl {y}"]   = SUPER_BOWL[y]

    return special


def _event_type(label: str) -> str:
    """Extract the event-type prefix from a label like 'Good Friday 2024' -> 'Good Friday'."""
    for et in _EVENT_TYPES:
        if label.startswith(et):
            return et
    # Fallback: strip trailing year
    return label.rsplit(" ", 1)[0]


# ---------------------------------------------------------------------------
# Data loader (cached — only reads the Excel once per Streamlit session)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_wi_sales() -> pd.DataFrame:
    """
    Load and clean the 'WI sales' sheet from the project Excel file.
    Header row is at index 5 (0-based) in the sheet.
    """
    xl_path = os.path.join(_PROJECT_ROOT, "20260202 Walk in sales - v1400.xlsx")

    df = pd.read_excel(xl_path, sheet_name="WI sales", header=5)

    # Drop columns that are entirely NaN
    df = df.dropna(axis=1, how="all")
    if "Loc number" in df.columns:
        df = df.drop(columns=["Loc number"])

    # Parse date — stored as string e.g. "2/8/2023" in this sheet
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Coerce numeric metric columns
    df["Unique Purchased Cards"] = pd.to_numeric(
        df["Unique Purchased Cards"], errors="coerce"
    )
    if "GL Rev" in df.columns:
        df["GL Rev"] = pd.to_numeric(df["GL Rev"], errors="coerce")

    # Extract integer loc number from "0010 - Ontario" -> 10
    extracted = df["Location"].str.extract(r"^0*(\d+)")
    df["Loc Number"] = pd.to_numeric(extracted[0], errors="coerce")
    df = df.dropna(subset=["Loc Number"])
    df["Loc Number"] = df["Loc Number"].astype(int)

    df = df[df["Loc Number"].isin(VALID_LOC_NUMBERS)]
    df = df.dropna(subset=["Unique Purchased Cards"])

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helper: location list sorted by total volume
# ---------------------------------------------------------------------------

def get_available_locs(df: pd.DataFrame) -> list:
    return (
        df.groupby("Loc Number")["Unique Purchased Cards"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_RESAMPLE_FREQ = {
    "Daily":     "D",
    "Weekly":    "W",
    "Monthly":   "ME",
    "Quarterly": "QE",
}


def aggregate_data(
    df: pd.DataFrame,
    interval: str,
    selected_locs: list,
    special_dates_map: dict,
    selected_event_types: list,
) -> pd.DataFrame:
    """
    Return a long-format DataFrame ready for Altair.
    For time-series intervals: Date, Location, Unique Purchased Cards.
    For Special Dates:         Event, Event Type, Location, Unique Purchased Cards.
    """
    sub = df[df["Loc Number"].isin(selected_locs)].copy()

    if interval == "Special Dates":
        return _agg_special_dates(sub, selected_locs, special_dates_map, selected_event_types)

    freq = _RESAMPLE_FREQ[interval]
    rows = []
    for loc_num in selected_locs:
        loc_df    = sub[sub["Loc Number"] == loc_num].set_index("Date")
        loc_label = LOC_LABELS.get(loc_num, str(loc_num))
        if loc_df.empty:
            continue
        grp = (
            loc_df.resample(freq)["Unique Purchased Cards"]
            .sum()
            .reset_index()
        )
        grp["Location"] = loc_label
        rows.append(grp)

    if not rows:
        return pd.DataFrame(columns=["Date", "Location", "Unique Purchased Cards"])

    return pd.concat(rows, ignore_index=True)


def _agg_special_dates(
    sub: pd.DataFrame,
    selected_locs: list,
    special_dates_map: dict,
    selected_event_types: list,
) -> pd.DataFrame:
    """
    Filter to selected special-event dates and return aggregated long-format data.
    Includes an 'Event Type' column so the chart can facet/color by event type.
    """
    sub = sub.copy()
    sub["Date"] = sub["Date"].dt.normalize()

    # Filter the special dates map down to only selected event types
    filtered_map = {
        label: d
        for label, d in special_dates_map.items()
        if _event_type(label) in selected_event_types
    }

    if not filtered_map:
        return pd.DataFrame(columns=["Event", "Event Type", "Location", "Unique Purchased Cards"])

    special_ts  = {label: pd.Timestamp(d) for label, d in filtered_map.items()}
    ts_to_label = {v: k for k, v in special_ts.items()}

    filtered = sub[sub["Date"].isin(set(special_ts.values()))].copy()
    filtered["Event"]      = filtered["Date"].map(ts_to_label)
    filtered["Event Type"] = filtered["Event"].apply(_event_type)

    result_rows = []
    for loc_num in selected_locs:
        loc_label    = LOC_LABELS.get(loc_num, str(loc_num))
        loc_filtered = filtered[filtered["Loc Number"] == loc_num]
        if loc_filtered.empty:
            continue
        grp = (
            loc_filtered.groupby(["Event", "Event Type"])["Unique Purchased Cards"]
            .sum()
            .reset_index()
        )
        grp["Location"] = loc_label
        result_rows.append(grp)

    if not result_rows:
        return pd.DataFrame(columns=["Event", "Event Type", "Location", "Unique Purchased Cards"])

    return pd.concat(result_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary_stats(df: pd.DataFrame, selected_locs: list) -> tuple:
    sub = df[df["Loc Number"].isin(selected_locs)].copy()

    if sub.empty:
        return (
            {"total_cards": 0, "avg_per_day": 0.0,
             "peak_date": "N/A", "peak_value": "N/A", "peak_location": "N/A"},
            pd.DataFrame(),
        )

    total_cards = sub["Unique Purchased Cards"].sum()
    avg_per_day = sub.groupby("Date")["Unique Purchased Cards"].sum().mean()
    peak_idx    = sub["Unique Purchased Cards"].idxmax()
    peak_row    = sub.loc[peak_idx]

    metrics = {
        "total_cards":   total_cards,
        "avg_per_day":   avg_per_day,
        "peak_date":     peak_row["Date"].strftime("%Y-%m-%d"),
        "peak_value":    peak_row["Unique Purchased Cards"],
        "peak_location": LOC_LABELS.get(int(peak_row["Loc Number"]), str(peak_row["Loc Number"])),
    }

    table_rows = []
    for loc_num in selected_locs:
        loc_label = LOC_LABELS.get(loc_num, str(loc_num))
        loc_df    = sub[sub["Loc Number"] == loc_num]
        if loc_df.empty:
            table_rows.append({"Location": loc_label, "Total Cards": 0,
                                "Avg / Day": 0.0, "Peak Day": "No data", "Peak Cards": 0})
            continue
        loc_daily    = loc_df.groupby("Date")["Unique Purchased Cards"].sum()
        loc_peak_idx = loc_df["Unique Purchased Cards"].idxmax()
        loc_peak_row = loc_df.loc[loc_peak_idx]
        table_rows.append({
            "Location":    loc_label,
            "Total Cards": int(loc_df["Unique Purchased Cards"].sum()),
            "Avg / Day":   round(float(loc_daily.mean()), 1),
            "Peak Day":    loc_peak_row["Date"].strftime("%Y-%m-%d"),
            "Peak Cards":  int(loc_peak_row["Unique Purchased Cards"]),
        })

    return metrics, pd.DataFrame(table_rows).set_index("Location")


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def make_line_chart(plot_df: pd.DataFrame) -> alt.Chart:
    """Interactive time-series line chart — one line per location."""
    return (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y(
                "Unique Purchased Cards:Q",
                title="Unique Purchased Cards",
                axis=alt.Axis(format=","),
            ),
            color=alt.Color("Location:N", legend=alt.Legend(title="Location")),
            tooltip=[
                alt.Tooltip("Date:T",                   title="Date",     format="%Y-%m-%d"),
                alt.Tooltip("Location:N",               title="Location"),
                alt.Tooltip("Unique Purchased Cards:Q", title="Cards",    format=","),
            ],
        )
        .properties(height=450)
        .interactive()
    )


def make_special_dates_chart(plot_df: pd.DataFrame, color_by: str) -> alt.Chart:
    """
    Grouped bar chart for Special Dates mode.
    X axis  = individual event label (e.g. 'Good Friday 2024')
    Y axis  = Unique Purchased Cards
    Color   = Location or Event Type (user-controlled)
    Bars are side-by-side (xOffset) so each location is visible separately.
    """
    color_field = f"{color_by}:N"
    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "Event:N",
                title="Event",
                sort=None,   # preserve chronological sort applied before passing
                axis=alt.Axis(labelAngle=-45, labelLimit=200),
            ),
            xOffset=alt.XOffset(f"{color_by}:N"),   # side-by-side grouping
            y=alt.Y(
                "Unique Purchased Cards:Q",
                title="Unique Purchased Cards",
                axis=alt.Axis(format=","),
            ),
            color=alt.Color(
                color_field,
                legend=alt.Legend(title=color_by),
            ),
            tooltip=[
                alt.Tooltip("Event:N",                  title="Event"),
                alt.Tooltip("Event Type:N",             title="Event Type"),
                alt.Tooltip("Location:N",               title="Location"),
                alt.Tooltip("Unique Purchased Cards:Q", title="Cards", format=","),
            ],
        )
        .properties(height=450)
    )


# ===========================================================================
# PAGE BODY
# ===========================================================================

st.title("Historical Data Analysis")
st.markdown(
    "Explore historical walk-in sales data across store locations. "
    "Select one or more locations, choose a time interval, and optionally "
    "filter to specific special events such as holidays and major shopping days."
)
st.divider()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with st.spinner("Loading WI Sales data — this may take up to 30 seconds on first load…"):
    df_raw = load_wi_sales()

if df_raw.empty:
    st.error(
        "No data could be loaded from the WI Sales sheet. "
        "Check that the Excel file exists at the project root."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Special dates map (computed once from the data's year range)
# ---------------------------------------------------------------------------

data_years     = sorted(df_raw["Date"].dt.year.unique().tolist())
analysis_years = [y for y in data_years if 2023 <= y <= 2025] or [2023, 2024, 2025]
SPECIAL_DATES_MAP = build_special_dates(analysis_years)

# ---------------------------------------------------------------------------
# Location multiselect
# ---------------------------------------------------------------------------

available_locs = get_available_locs(df_raw)
default_locs   = available_locs[:3]

selected_locs = st.multiselect(
    "Select Locations",
    options=available_locs,
    default=default_locs,
    format_func=lambda n: LOC_LABELS.get(n, str(n)),
    help="Choose one or more store locations. Defaults to the top 3 by total card volume.",
)

if not selected_locs:
    st.warning("Please select at least one location to display the chart.", icon="⚠️")
    st.stop()

# ---------------------------------------------------------------------------
# Interval + date-range / event-type controls
# ---------------------------------------------------------------------------

col_interval, col_right = st.columns([1, 2])

with col_interval:
    interval = st.radio(
        "Time Interval",
        options=["Daily", "Weekly", "Monthly", "Quarterly", "Special Dates"],
        index=2,
        help=(
            "**Daily** — one point per day  \n"
            "**Weekly / Monthly / Quarterly** — summed over the period  \n"
            "**Special Dates** — filter to selected holidays & key events"
        ),
    )

with col_right:
    if interval == "Special Dates":
        # Event-type filter — ALL selected by default, user can narrow down
        selected_event_types = st.multiselect(
            "Filter by Event Type",
            options=_EVENT_TYPES,
            default=_EVENT_TYPES,
            help=(
                "Choose which categories of special days to display. "
                "Deselect all others to focus on a single event type across years."
            ),
        )

        # Color-by toggle: color bars by Location or by Event Type
        color_by = st.radio(
            "Color bars by",
            options=["Location", "Event Type"],
            index=0,
            horizontal=True,
            help=(
                "**Location** — compare the same event across selected stores.  \n"
                "**Event Type** — compare different event types within a store."
            ),
        )

        df_filtered = df_raw   # Special Dates uses all dates; filtering is done in aggregation

        n_events = sum(
            1 for lbl in SPECIAL_DATES_MAP
            if _event_type(lbl) in selected_event_types
        )
        st.info(
            f"Showing **{n_events} event instance(s)** "
            f"across {', '.join(str(y) for y in analysis_years)}.",
            icon="📅",
        )
    else:
        selected_event_types = _EVENT_TYPES   # unused for non-special modes
        color_by = "Location"

        min_date   = df_raw["Date"].min().date()
        max_date   = df_raw["Date"].max().date()
        date_range = st.slider(
            "Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            help="Filter the data to this date range before aggregating.",
        )
        df_filtered = df_raw[
            (df_raw["Date"] >= pd.Timestamp(date_range[0])) &
            (df_raw["Date"] <= pd.Timestamp(date_range[1]))
        ]

# ---------------------------------------------------------------------------
# Guard: need at least one event type selected in Special Dates mode
# ---------------------------------------------------------------------------

if interval == "Special Dates" and not selected_event_types:
    st.warning("Please select at least one event type to display the chart.", icon="⚠️")
    st.stop()

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

plot_df = aggregate_data(
    df_filtered, interval, selected_locs, SPECIAL_DATES_MAP, selected_event_types
)

if plot_df.empty:
    st.warning(
        "No data found for the selected locations, date range, and event types. "
        "Try broadening your filters.",
        icon="⚠️",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

if interval == "Special Dates":
    # Sort events chronologically so the X axis reads left-to-right in time
    event_order = sorted(
        plot_df["Event"].unique(),
        key=lambda e: SPECIAL_DATES_MAP.get(e, datetime.date.min),
    )
    plot_df["Event"] = pd.Categorical(plot_df["Event"], categories=event_order, ordered=True)
    plot_df = plot_df.sort_values("Event")
    chart = make_special_dates_chart(plot_df, color_by)
else:
    chart = make_line_chart(plot_df)

st.altair_chart(chart, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

metrics, per_loc_table = compute_summary_stats(df_filtered, selected_locs)

st.subheader("Summary Statistics")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Cards", f"{int(metrics['total_cards']):,}")
with m2:
    st.metric("Avg Cards / Day", f"{metrics['avg_per_day']:,.1f}")
with m3:
    st.metric("Peak Day", str(metrics["peak_date"]))
with m4:
    if metrics["peak_value"] != "N/A":
        st.metric(
            "Peak Value",
            f"{int(metrics['peak_value']):,} cards",
            help=f"At {metrics['peak_location']}",
        )
    else:
        st.metric("Peak Value", "N/A")

if not per_loc_table.empty:
    st.markdown("#### Per-Location Breakdown")
    st.dataframe(
        per_loc_table,
        use_container_width=True,
        column_config={
            "Total Cards": st.column_config.NumberColumn(format="%d"),
            "Avg / Day":   st.column_config.NumberColumn(format="%.1f"),
            "Peak Cards":  st.column_config.NumberColumn(format="%d"),
        },
    )
