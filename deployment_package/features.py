"""
features.py
-----------
Single source of truth for constants and feature engineering.
Imported by app.py, model_registry.py, and train_models.py.
"""

from __future__ import annotations
from typing import Any

import pandas as pd
import numpy as np
import holidays

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURES = [
    "Precipitation",
    "DayOfWeek",
    "Month",
    "DayOfMonth",
    "WeekOfYear",
    "IsWeekend",
    "Quarter",
    "Loc Number",
]

TARGET = "GL Rev"
TARGET_CARDS = "Unique Purchased Cards"

# All unique Loc Numbers present in the training data (sorted ascending)
VALID_LOC_NUMBERS = [
    4, 5, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 42, 43, 44, 45, 47,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66,
    67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
    85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
    102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
    116, 117, 118, 119, 120, 121, 122, 124, 125, 126, 127, 128, 129, 130,
    131, 132, 133, 134, 135, 136, 137, 138, 140, 141, 142, 143, 144, 145,
    146, 147, 148, 150, 151, 152, 153, 156, 157, 158, 159, 160, 161, 162,
    163, 164, 165, 166, 167, 168, 169, 170, 177, 179, 189,
]

MAX_PRECIPITATION = 782.49      # max value seen during training (old precipitation)
MAX_PRECIPITATION_NEW = 7.71   # max value seen during new-precip model training (inches)

# V2 features — adds holiday awareness on top of the original set
FEATURES_V2 = [
    "Precipitation",
    "DayOfWeek",
    "Month",
    "DayOfMonth",
    "WeekOfYear",
    "IsWeekend",
    "Quarter",
    "Loc Number",
    "IsHoliday",
    "IsHolidayWeekend",
    "DaysToNearestHoliday",
]

# V3 features — V2 + lag/rolling features for short-term forecasting
FEATURES_V3 = [
    "Precipitation",
    "DayOfWeek",
    "Month",
    "DayOfMonth",
    "WeekOfYear",
    "IsWeekend",
    "Quarter",
    "Loc Number",
    "IsHoliday",
    "IsHolidayWeekend",
    "DaysToNearestHoliday",
    "Lag7",
    "Lag14",
    "Roll7Mean",
    "Roll14Mean",
]

# ---------------------------------------------------------------------------
# Location display labels  (Loc Number -> "N - City")
# Sourced from the 'Location Lookup' sheet in the data file.
# Loc 189 has no entry in that sheet and falls back to the number only.
# ---------------------------------------------------------------------------

LOC_LABELS: dict[int, str] = {
    4: "4 - Atlanta",
    5: "5 - Philadelphia",
    8: "8 - Hollywood",
    10: "10 - Ontario",
    11: "11 - Cincinnati",
    12: "12 - Denver",
    13: "13 - Utica",
    14: "14 - Irvine",
    15: "15 - Palisades",
    16: "16 - Orange",
    17: "17 - Hilliard",
    18: "18 - San Antonio",
    20: "20 - St. Louis",
    21: "21 - Austin",
    22: "22 - Jacksonville",
    23: "23 - Providence",
    24: "24 - San Jose",
    25: "25 - Westminster",
    26: "26 - Pittsburgh",
    27: "27 - San Diego",
    28: "28 - Miami",
    29: "29 - Frisco",
    31: "31 - Cleveland",
    32: "32 - Islandia",
    33: "33 - Toronto",
    34: "34 - Santa Anita",
    35: "35 - Arundel",
    36: "36 - Concord",
    38: "38 - Franklin",
    39: "39 - Houston II",
    42: "42 - Nashville",
    43: "43 - Scottsdale",
    44: "44 - Westbury",
    45: "45 - Lawrenceville",
    47: "47 - Omaha",
    49: "49 - Kansas City",
    50: "50 - TimesSquare",
    51: "51 - Maple Grove",
    52: "52 - Tempe",
    53: "53 - Plymouth Meeting",
    54: "54 - Arlington",
    55: "55 - Richmond",
    56: "56 - Tulsa",
    57: "57 - Indianapolis",
    58: "58 - Polaris",
    59: "59 - Wauwatosa",
    60: "60 - Roseville",
    61: "61 - Braintree",
    62: "62 - Dallas",
    63: "63 - Clackamas",
    64: "64 - Orlando",
    66: "66 - Oklahoma City",
    67: "67 - Orland Park",
    68: "68 - Boise",
    69: "69 - Virginia Beach",
    70: "70 - Albany",
    71: "71 - Syracuse",
    72: "72 - Greenville",
    74: "74 - Livonia",
    75: "75 - Westchester",
    76: "76 - Vernon Hills",
    77: "77 - Panama City",
    78: "78 - Los Angeles",
    79: "79 - Albuquerque",
    80: "80 - Manchester",
    81: "81 - Euless",
    82: "82 - Pelham",
    83: "83 - Rivercenter",
    84: "84 - Woburn",
    85: "85 - Kentwood",
    86: "86 - Buffalo-Walden",
    87: "87 - Edina",
    88: "88 - Fresno",
    89: "89 - Friendswood",
    90: "90 - Glendale",
    91: "91 - Springfield",
    92: "92 - El Paso",
    93: "93 - Rochester",
    94: "94 - Summerlin",
    95: "95 - Capitol Heights",
    96: "96 - Florence",
    97: "97 - Little Rock",
    98: "98 - Oakville",
    99: "99 - Silver Spring",
    100: "100 - Toledo",
    101: "101 - Overland Park",
    102: "102 - Daly City",
    103: "103 - Carlsbad",
    104: "104 - Columbia",
    105: "105 - Tucson",
    106: "106 - New Orleans",
    107: "107 - Myrtle Beach",
    108: "108 - Alpharetta",
    109: "109 - McAllen",
    110: "110 - Anchorage",
    111: "111 - Pineville",
    112: "112 - Northridge",
    113: "113 - Bayamon",
    114: "114 - Wayne",
    115: "115 - Auburn",
    116: "116 - Baltimore",
    117: "117 - Rogers",
    118: "118 - Woodbridge",
    119: "119 - Memphis",
    120: "120 - Tampa",
    121: "121 - Madison",
    122: "122 - Torrance",
    124: "124 - Salt Lake City",
    125: "125 - Milford",
    126: "126 - Rosemont",
    127: "127 - Thousand Oaks",
    128: "128 - Birmingham",
    129: "129 - Fairfax",
    130: "130 - Staten Island",
    131: "131 - Louisville",
    132: "132 - Harrisburg",
    133: "133 - Corpus Christi",
    134: "134 - North Hills",
    135: "135 - Daytona Beach",
    136: "136 - Fort Myers",
    137: "137 - Sevierville",
    138: "138 - Winston-Salem",
    140: "140 - McDonough",
    141: "141 - Gaithersburg",
    142: "142 - Shenandoah",
    143: "143 - Natick",
    144: "144 - Huntsville",
    145: "145 - Wichita",
    146: "146 - Canton",
    147: "147 - New Hampshire",
    148: "148 - Modesto",
    150: "150 - Bellevue",
    151: "151 - Gloucester",
    152: "152 - Fairfield",
    153: "153 - Long Beach",
    156: "156 - Greenwood",
    157: "157 - Concord CA",
    158: "158 - Lynnwood",
    159: "159 - Chattanooga",
    160: "160 - Lehigh Valley",
    161: "161 - Green Bay",
    162: "162 - Sioux Falls",
    163: "163 - Brooklyn Gateway",
    164: "164 - Bakersfield",
    165: "165 - Gainesville",
    166: "166 - Brooklyn Atlantic",
    167: "167 - San Juan",
    168: "168 - Augusta",
    169: "169 - Cary II",
    170: "170 - Lubbock",
    177: "177 - Des Moines",
    179: "179 - Queen Creek",
    189: "189 - Unknown",
}

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all calendar-derived columns to a copy of df."""
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out["DayOfWeek"]  = out["Date"].dt.dayofweek
    out["Month"]      = out["Date"].dt.month
    out["DayOfMonth"] = out["Date"].dt.day
    out["WeekOfYear"] = out["Date"].dt.isocalendar().week.astype(int)
    out["IsWeekend"]  = (out["DayOfWeek"] >= 5).astype(int)
    out["Quarter"]    = out["Date"].dt.quarter
    return out


def _add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add holiday-awareness columns to a DataFrame that already has a
    datetime 'Date' column.

    New columns:
      IsHoliday           - 1 if the date is a US federal holiday, else 0
      IsHolidayWeekend    - 1 if the date falls within a holiday weekend
                            (the Sat-Sun-Mon cluster around a Monday holiday,
                             or the Fri-Sat-Sun cluster around a Friday holiday,
                             or any day within 1 day of any holiday)
      DaysToNearestHoliday - integer distance (0-183) to the closest holiday
    """
    out = df.copy()
    dates = pd.to_datetime(out["Date"])

    # Build a set of US holiday dates covering all years in the data + 1 margin
    years = sorted(dates.dt.year.unique())
    year_range = range(min(years) - 1, max(years) + 2)
    us_holidays = holidays.US(years=year_range)
    holiday_dates = np.array(sorted(us_holidays.keys()), dtype="datetime64[D]")

    date_vals = dates.values.astype("datetime64[D]")

    # IsHoliday: exact match
    holiday_set = set(us_holidays.keys())
    out["IsHoliday"] = dates.dt.date.map(lambda d: 1 if d in holiday_set else 0)

    # DaysToNearestHoliday: vectorized distance to closest holiday
    # For each date, find the minimum absolute distance to any holiday
    date_int = date_vals.astype(np.int64)
    hol_int = holiday_dates.astype(np.int64)
    # Use searchsorted for efficient nearest-neighbor lookup
    idx = np.searchsorted(hol_int, date_int, side="left")
    idx = np.clip(idx, 1, len(hol_int) - 1)
    dist_left = np.abs(date_int - hol_int[idx - 1])
    dist_right = np.abs(date_int - hol_int[idx])
    out["DaysToNearestHoliday"] = np.minimum(dist_left, dist_right).astype(int)

    # IsHolidayWeekend: date is within 1 day of a holiday
    # This captures Fri/Sat/Sun/Mon around holiday weekends
    out["IsHolidayWeekend"] = (out["DaysToNearestHoliday"] <= 1).astype(int)

    return out


def _add_lag_features(
    df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
    target_col: str = "Unique Purchased Cards",
) -> pd.DataFrame:
    """
    Add lag and rolling-window features per location.

    Parameters
    ----------
    df : DataFrame
        Must have columns: Loc Number, Date.
        If used during training, must also have `target_col`.
    history_df : DataFrame or None
        At inference time, a DataFrame with columns
        [Loc Number, Date, <target_col>] containing recent historical
        sales.  At training time, pass None (lags come from df itself).
    target_col : str
        Name of the column with actual sales values.

    New columns:
      Lag7       - sales exactly 7 calendar days ago at the same location
      Lag14      - sales exactly 14 calendar days ago at the same location
      Roll7Mean  - mean of past 7 days (shift-1, per location)
      Roll14Mean - mean of past 14 days (shift-1, per location)

    Missing lags are filled with the location's day-of-week median,
    then with the global median as a final fallback.
    """
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])

    # --- Build the source data for lag computation -------------------------
    if history_df is not None:
        # INFERENCE PATH: merge prediction rows with history
        hist = history_df.copy()
        hist["Date"] = pd.to_datetime(hist["Date"])
        combined = pd.concat([
            hist[["Loc Number", "Date", target_col]],
            out[["Loc Number", "Date"]].assign(**{target_col: np.nan}),
        ], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["Loc Number", "Date"], keep="first"
        )
    else:
        # TRAINING PATH: target is already in df
        combined = out[["Loc Number", "Date", target_col]].copy()

    combined = combined.sort_values(["Loc Number", "Date"]).reset_index(drop=True)

    # --- Lag7 / Lag14: merge-based (exact calendar-day semantics) ----------
    for lag_days, col_name in [(7, "Lag7"), (14, "Lag14")]:
        lookup = combined[["Loc Number", "Date", target_col]].copy()
        lookup["Date"] = lookup["Date"] + pd.Timedelta(days=lag_days)
        lookup = lookup.rename(columns={target_col: col_name})
        combined = combined.merge(
            lookup[["Loc Number", "Date", col_name]],
            on=["Loc Number", "Date"],
            how="left",
        )

    # --- Roll7Mean / Roll14Mean: shift-based rolling per location ----------
    combined = combined.sort_values(["Loc Number", "Date"]).reset_index(drop=True)
    grouped = combined.groupby("Loc Number", observed=True)[target_col]
    combined["Roll7Mean"] = grouped.transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=1).mean()
    )
    combined["Roll14Mean"] = grouped.transform(
        lambda s: s.shift(1).rolling(window=14, min_periods=1).mean()
    )

    # --- Fallback: per-location day-of-week median -------------------------
    combined["_dow"] = combined["Date"].dt.dayofweek
    fallback = (
        combined.dropna(subset=[target_col])
        .groupby(["Loc Number", "_dow"], observed=True)[target_col]
        .median()
        .rename("_fallback")
    )
    combined = combined.merge(fallback, on=["Loc Number", "_dow"], how="left")

    lag_cols = ["Lag7", "Lag14", "Roll7Mean", "Roll14Mean"]
    global_median = combined[target_col].median()
    for col in lag_cols:
        combined[col] = combined[col].fillna(combined["_fallback"])
        combined[col] = combined[col].fillna(global_median)

    combined = combined.drop(columns=["_dow", "_fallback"])

    # --- Merge lag columns back onto original rows -------------------------
    merge_cols = ["Loc Number", "Date"] + lag_cols
    out = out.merge(
        combined[merge_cols].drop_duplicates(subset=["Loc Number", "Date"]),
        on=["Loc Number", "Date"],
        how="left",
    )

    return out


def _validate_loc_numbers(df: pd.DataFrame) -> None:
    """Raise ValueError if any Loc Number was not seen during training."""
    invalid = set(df["Loc Number"].unique()) - set(VALID_LOC_NUMBERS)
    if invalid:
        raise ValueError(
            f"Loc Number value(s) not seen during training: {sorted(invalid)}. "
            f"Valid values are integers in: {VALID_LOC_NUMBERS}"
        )

# ---------------------------------------------------------------------------
# Public feature engineering functions
# ---------------------------------------------------------------------------

def engineer_features_lgbm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for LightGBM.
    Loc Number is encoded as pd.Categorical with explicit categories matching
    the training-time encoding, so category codes are identical at inference.

    Input:  DataFrame with columns [Loc Number (int), Date (datetime-like), Precipitation (float)]
    Output: DataFrame with exactly FEATURES columns in correct order and dtypes.
    """
    out = _add_calendar_features(df)
    _validate_loc_numbers(out)
    out["Loc Number"] = pd.Categorical(
        out["Loc Number"],
        categories=VALID_LOC_NUMBERS,
    )
    return out[FEATURES]


def engineer_features_lgbm_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    V2 feature engineering for LightGBM — adds holiday features.

    Same as engineer_features_lgbm but includes:
      IsHoliday, IsHolidayWeekend, DaysToNearestHoliday

    Input:  DataFrame with columns [Loc Number (int), Date (datetime-like), Precipitation (float)]
    Output: DataFrame with exactly FEATURES_V2 columns in correct order and dtypes.
    """
    out = _add_calendar_features(df)
    out = _add_holiday_features(out)
    _validate_loc_numbers(out)
    out["Loc Number"] = pd.Categorical(
        out["Loc Number"],
        categories=VALID_LOC_NUMBERS,
    )
    return out[FEATURES_V2]


def engineer_features_lgbm_v3(
    df: pd.DataFrame,
    model_bundle: Any = None,
) -> pd.DataFrame:
    """
    V3 feature engineering for LightGBM — holidays + lag/rolling features.

    At inference time, ``model_bundle`` is the tuple returned by the V3
    load function: ``(booster, calibrator, history_df)``.  The history_df
    is used to compute lag features.

    At training time, call with ``model_bundle=None``; lags are computed
    from ``df`` itself (which must contain the target column).

    Input:  DataFrame with columns [Loc Number, Date, Precipitation]
            (and the target column when training).
    Output: DataFrame with exactly FEATURES_V3 columns.
    """
    out = _add_calendar_features(df)
    out = _add_holiday_features(out)

    # Extract history from model bundle (inference) or use None (training)
    history_df = None
    if model_bundle is not None:
        _, _, history_df = model_bundle  # (booster, calibrator, history)

    out = _add_lag_features(out, history_df=history_df)

    _validate_loc_numbers(out)
    out["Loc Number"] = pd.Categorical(
        out["Loc Number"],
        categories=VALID_LOC_NUMBERS,
    )
    return out[FEATURES_V3]


def engineer_features_sklearn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for scikit-learn models.
    Loc Number is kept as plain int; the Pipeline's ColumnTransformer +
    OneHotEncoder (fit with categories=VALID_LOC_NUMBERS) handles encoding.

    Input:  DataFrame with columns [Loc Number (int), Date (datetime-like), Precipitation (float)]
    Output: DataFrame with exactly FEATURES columns in correct order and dtypes.
    """
    out = _add_calendar_features(df)
    _validate_loc_numbers(out)
    out["Loc Number"] = out["Loc Number"].astype(int)
    return out[FEATURES]
