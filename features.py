"""
features.py
-----------
Single source of truth for constants and feature engineering.
Imported by app.py, model_registry.py, and train_models.py.
"""

import pandas as pd

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

MAX_PRECIPITATION = 782.49  # max value seen during training

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
