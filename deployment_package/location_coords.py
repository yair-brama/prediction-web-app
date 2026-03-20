"""
location_coords.py
------------------
Lookup table mapping Loc Number (int) -> geographic coordinates.

HOW TO FILL THIS IN
-------------------
1. Replace every placeholder 0.0 / 0.0 pair with the actual latitude and
   longitude of that store location.

2. Coordinates must be in decimal degrees:
     Latitude  : positive = North, negative = South  (e.g. 40.7128 for NYC)
     Longitude : positive = East,  negative = West   (US stores are negative,
                                                       e.g. -74.0060 for NYC)

3. The "name" field is optional but helps with debugging and UI display.
   Replace "Store N" with a human-readable label if you have one.

4. RECOMMENDED ways to find coordinates:
     - Google Maps : right-click any point on the map -> the first line shown
                     is "lat, lon" (e.g. "40.7128, -74.0060")
     - Your internal store / HR database
     - A geocoding API (geopy, googlemaps, positionstack) if you only have
       street addresses

5. Stores whose coordinates are still 0.0 / 0.0 will show a warning in the
   app when weather auto-fill is attempted. They do NOT break the app.

6. Do NOT commit real store addresses or PII to a public repository.
"""

# Sentinel values used to detect un-filled placeholders.
# (0.0, 0.0) is in the Gulf of Guinea -- no US retail store is there.
# Do NOT change these values.
_PLACEHOLDER_LAT = 0.0
_PLACEHOLDER_LON = 0.0


LOCATION_COORDS: dict[int, dict] = {
    # <Loc Number>: {"lat": <float>, "lon": <float>, "name": "<str>"},
    # Coordinates geocoded via Open-Meteo geocoding API (geocoding-api.open-meteo.com).

    4:   {"lat": 33.74900, "lon": -84.38798, "name": "Atlanta, GA"},
    5:   {"lat": 39.95238, "lon": -75.16362, "name": "Philadelphia, PA"},
    8:   {"lat": 26.01120, "lon": -80.14949, "name": "Hollywood, FL"},
    10:  {"lat": 34.06334, "lon": -117.65089, "name": "Ontario, CA"},
    11:  {"lat": 39.12711, "lon": -84.51439, "name": "Cincinnati, OH"},
    12:  {"lat": 39.73915, "lon": -104.98470, "name": "Denver, CO"},
    13:  {"lat": 43.10090, "lon": -75.23266, "name": "Utica, NY"},
    14:  {"lat": 33.66946, "lon": -117.82311, "name": "Irvine, CA"},
    15:  {"lat": 40.84816, "lon": -73.99764, "name": "Palisades Park, NJ"},
    16:  {"lat": 33.78779, "lon": -117.85311, "name": "Orange, CA"},
    17:  {"lat": 40.03340, "lon": -83.15825, "name": "Hilliard, OH"},
    18:  {"lat": 29.42412, "lon": -98.49363, "name": "San Antonio, TX"},
    20:  {"lat": 38.62727, "lon": -90.19789, "name": "Saint Louis, MO"},
    21:  {"lat": 30.26715, "lon": -97.74306, "name": "Austin, TX"},
    22:  {"lat": 30.33218, "lon": -81.65565, "name": "Jacksonville, FL"},
    23:  {"lat": 41.82399, "lon": -71.41283, "name": "Providence, RI"},
    24:  {"lat": 37.33939, "lon": -121.89496, "name": "San Jose, CA"},
    25:  {"lat": 39.83665, "lon": -105.03720, "name": "Westminster, CO"},
    26:  {"lat": 40.44062, "lon": -79.99589, "name": "Pittsburgh, PA"},
    27:  {"lat": 32.71571, "lon": -117.16472, "name": "San Diego, CA"},
    28:  {"lat": 25.77427, "lon": -80.19366, "name": "Miami, FL"},
    29:  {"lat": 33.15067, "lon": -96.82361, "name": "Frisco, TX"},
    31:  {"lat": 41.49950, "lon": -81.69541, "name": "Cleveland, OH"},
    32:  {"lat": 40.80426, "lon": -73.16900, "name": "Islandia, NY"},
    33:  {"lat": 43.70643, "lon": -79.39864, "name": "Toronto, ON"},
    34:  {"lat": 34.13973, "lon": -118.03534, "name": "Arcadia, CA"},
    35:  {"lat": 39.19344, "lon": -76.72442, "name": "Hanover, MD"},
    36:  {"lat": 35.40888, "lon": -80.58158, "name": "Concord, NC"},
    38:  {"lat": 35.92506, "lon": -86.86889, "name": "Franklin, TN"},
    39:  {"lat": 29.76328, "lon": -95.36327, "name": "Houston, TX"},
    42:  {"lat": 36.16589, "lon": -86.78444, "name": "Nashville, TN"},
    43:  {"lat": 33.50921, "lon": -111.89903, "name": "Scottsdale, AZ"},
    44:  {"lat": 40.75566, "lon": -73.58763, "name": "Westbury, NY"},
    45:  {"lat": 33.95621, "lon": -83.98796, "name": "Lawrenceville, GA"},
    47:  {"lat": 41.25626, "lon": -95.94043, "name": "Omaha, NE"},
    49:  {"lat": 39.09973, "lon": -94.57857, "name": "Kansas City, MO"},
    50:  {"lat": 40.71427, "lon": -74.00597, "name": "New York, NY"},
    51:  {"lat": 45.07246, "lon": -93.45579, "name": "Maple Grove, MN"},
    52:  {"lat": 33.41477, "lon": -111.90931, "name": "Tempe, AZ"},
    53:  {"lat": 40.10233, "lon": -75.27435, "name": "Plymouth Meeting, PA"},
    54:  {"lat": 32.73569, "lon": -97.10807, "name": "Arlington, TX"},
    55:  {"lat": 37.55376, "lon": -77.46026, "name": "Richmond, VA"},
    56:  {"lat": 36.15398, "lon": -95.99277, "name": "Tulsa, OK"},
    57:  {"lat": 39.76838, "lon": -86.15804, "name": "Indianapolis, IN"},
    58:  {"lat": 39.96118, "lon": -82.99879, "name": "Columbus, OH"},
    59:  {"lat": 43.04946, "lon": -88.00759, "name": "Wauwatosa, WI"},
    60:  {"lat": 38.75212, "lon": -121.28801, "name": "Roseville, CA"},
    61:  {"lat": 42.25288, "lon": -71.00227, "name": "Braintree, MA"},
    62:  {"lat": 32.78306, "lon": -96.80667, "name": "Dallas, TX"},
    63:  {"lat": 45.40762, "lon": -122.57037, "name": "Clackamas, OR"},
    64:  {"lat": 28.53834, "lon": -81.37924, "name": "Orlando, FL"},
    66:  {"lat": 35.46756, "lon": -97.51643, "name": "Oklahoma City, OK"},
    67:  {"lat": 41.63031, "lon": -87.85394, "name": "Orland Park, IL"},
    68:  {"lat": 43.61350, "lon": -116.20345, "name": "Boise, ID"},
    69:  {"lat": 36.85293, "lon": -75.97799, "name": "Virginia Beach, VA"},
    70:  {"lat": 42.65258, "lon": -73.75623, "name": "Albany, NY"},
    71:  {"lat": 43.04812, "lon": -76.14742, "name": "Syracuse, NY"},
    72:  {"lat": 34.85262, "lon": -82.39401, "name": "Greenville, SC"},
    74:  {"lat": 42.36837, "lon": -83.35271, "name": "Livonia, MI"},
    75:  {"lat": 33.96030, "lon": -118.43090, "name": "Westchester, CA"},
    76:  {"lat": 42.21947, "lon": -87.97952, "name": "Vernon Hills, IL"},
    77:  {"lat": 30.15946, "lon": -85.65983, "name": "Panama City, FL"},
    78:  {"lat": 34.05223, "lon": -118.24368, "name": "Los Angeles, CA"},
    79:  {"lat": 35.08449, "lon": -106.65114, "name": "Albuquerque, NM"},
    80:  {"lat": 42.99564, "lon": -71.45479, "name": "Manchester, NH"},
    81:  {"lat": 32.83707, "lon": -97.08195, "name": "Euless, TX"},
    82:  {"lat": 40.90982, "lon": -73.80791, "name": "Pelham, NY"},
    83:  {"lat": 29.42412, "lon": -98.49363, "name": "San Antonio, TX"},
    84:  {"lat": 42.47926, "lon": -71.15228, "name": "Woburn, MA"},
    85:  {"lat": 42.86947, "lon": -85.64475, "name": "Kentwood, MI"},
    86:  {"lat": 42.88645, "lon": -78.87837, "name": "Buffalo, NY"},
    87:  {"lat": 44.88969, "lon": -93.34995, "name": "Edina, MN"},
    88:  {"lat": 36.74773, "lon": -119.77237, "name": "Fresno, CA"},
    89:  {"lat": 29.52940, "lon": -95.20104, "name": "Friendswood, TX"},
    90:  {"lat": 33.53865, "lon": -112.18599, "name": "Glendale, AZ"},
    91:  {"lat": 38.78953, "lon": -77.18720, "name": "Springfield, VA"},
    92:  {"lat": 31.75872, "lon": -106.48693, "name": "El Paso, TX"},
    93:  {"lat": 43.15478, "lon": -77.61556, "name": "Rochester, NY"},
    94:  {"lat": 36.17497, "lon": -115.13722, "name": "Las Vegas, NV"},
    95:  {"lat": 38.88511, "lon": -76.91581, "name": "Capitol Heights, MD"},
    96:  {"lat": 34.19543, "lon": -79.76256, "name": "Florence, SC"},
    97:  {"lat": 34.74648, "lon": -92.28959, "name": "Little Rock, AR"},
    98:  {"lat": 43.45011, "lon": -79.68292, "name": "Oakville, ON"},
    99:  {"lat": 38.99067, "lon": -77.02609, "name": "Silver Spring, MD"},
    100: {"lat": 41.66394, "lon": -83.55521, "name": "Toledo, OH"},
    101: {"lat": 38.98223, "lon": -94.67079, "name": "Overland Park, KS"},
    102: {"lat": 37.70577, "lon": -122.46192, "name": "Daly City, CA"},
    103: {"lat": 33.15809, "lon": -117.35059, "name": "Carlsbad, CA"},
    104: {"lat": 34.00071, "lon": -81.03481, "name": "Columbia, SC"},
    105: {"lat": 32.22174, "lon": -110.92648, "name": "Tucson, AZ"},
    106: {"lat": 29.95465, "lon": -90.07507, "name": "New Orleans, LA"},
    107: {"lat": 33.68906, "lon": -78.88669, "name": "Myrtle Beach, SC"},
    108: {"lat": 34.07538, "lon": -84.29409, "name": "Alpharetta, GA"},
    109: {"lat": 26.20341, "lon": -98.23001, "name": "McAllen, TX"},
    110: {"lat": 61.21806, "lon": -149.90028, "name": "Anchorage, AK"},
    111: {"lat": 35.08320, "lon": -80.89230, "name": "Pineville, NC"},
    112: {"lat": 34.22834, "lon": -118.53675, "name": "Northridge, CA"},
    113: {"lat": 18.38079, "lon": -66.15328, "name": "Bayamon, PR"},
    114: {"lat": 40.92538, "lon": -74.27654, "name": "Wayne, NJ"},
    115: {"lat": 47.30732, "lon": -122.22845, "name": "Auburn, WA"},
    116: {"lat": 39.29038, "lon": -76.61219, "name": "Baltimore, MD"},
    117: {"lat": 36.33202, "lon": -94.11854, "name": "Rogers, AR"},
    118: {"lat": 40.55760, "lon": -74.28459, "name": "Woodbridge, NJ"},
    119: {"lat": 35.14953, "lon": -90.04898, "name": "Memphis, TN"},
    120: {"lat": 27.94752, "lon": -82.45843, "name": "Tampa, FL"},
    121: {"lat": 43.07305, "lon": -89.40123, "name": "Madison, WI"},
    122: {"lat": 33.83585, "lon": -118.34063, "name": "Torrance, CA"},
    124: {"lat": 40.76078, "lon": -111.89105, "name": "Salt Lake City, UT"},
    125: {"lat": 41.22232, "lon": -73.05650, "name": "Milford, CT"},
    126: {"lat": 41.99531, "lon": -87.88451, "name": "Rosemont, IL"},
    127: {"lat": 34.17056, "lon": -118.83759, "name": "Thousand Oaks, CA"},
    128: {"lat": 33.52066, "lon": -86.80249, "name": "Birmingham, AL"},
    129: {"lat": 38.84622, "lon": -77.30637, "name": "Fairfax, VA"},
    130: {"lat": 40.56233, "lon": -74.13986, "name": "Staten Island, NY"},
    131: {"lat": 38.25424, "lon": -85.75941, "name": "Louisville, KY"},
    132: {"lat": 40.27370, "lon": -76.88442, "name": "Harrisburg, PA"},
    133: {"lat": 27.80058, "lon": -97.39638, "name": "Corpus Christi, TX"},
    134: {"lat": 40.11380, "lon": -75.07100, "name": "North Hills, PA"},
    135: {"lat": 29.21081, "lon": -81.02283, "name": "Daytona Beach, FL"},
    136: {"lat": 26.62168, "lon": -81.84059, "name": "Fort Myers, FL"},
    137: {"lat": 35.86815, "lon": -83.56184, "name": "Sevierville, TN"},
    138: {"lat": 36.09986, "lon": -80.24422, "name": "Winston-Salem, NC"},
    140: {"lat": 33.44734, "lon": -84.14686, "name": "McDonough, GA"},
    141: {"lat": 39.14344, "lon": -77.20137, "name": "Gaithersburg, MD"},
    142: {"lat": 30.18022, "lon": -95.45577, "name": "Shenandoah, TX"},
    143: {"lat": 42.28343, "lon": -71.34950, "name": "Natick, MA"},
    144: {"lat": 34.73040, "lon": -86.58594, "name": "Huntsville, AL"},
    145: {"lat": 37.69224, "lon": -97.33754, "name": "Wichita, KS"},
    146: {"lat": 40.79895, "lon": -81.37845, "name": "Canton, OH"},
    147: {"lat": 42.99564, "lon": -71.45479, "name": "Manchester, NH"},
    148: {"lat": 37.63910, "lon": -120.99688, "name": "Modesto, CA"},
    150: {"lat": 47.61038, "lon": -122.20068, "name": "Bellevue, WA"},
    151: {"lat": 39.89281, "lon": -75.12041, "name": "Gloucester City, NJ"},
    152: {"lat": 38.24936, "lon": -122.03997, "name": "Fairfield, CA"},
    153: {"lat": 33.76696, "lon": -118.18923, "name": "Long Beach, CA"},
    156: {"lat": 39.61366, "lon": -86.10665, "name": "Greenwood, IN"},
    157: {"lat": 37.97798, "lon": -122.03107, "name": "Concord, CA"},
    158: {"lat": 47.82093, "lon": -122.31513, "name": "Lynnwood, WA"},
    159: {"lat": 35.04563, "lon": -85.30968, "name": "Chattanooga, TN"},
    160: {"lat": 40.60843, "lon": -75.49018, "name": "Allentown, PA"},
    161: {"lat": 44.51916, "lon": -88.01983, "name": "Green Bay, WI"},
    162: {"lat": 43.54997, "lon": -96.70033, "name": "Sioux Falls, SD"},
    163: {"lat": 40.65010, "lon": -73.94958, "name": "Brooklyn, NY"},
    164: {"lat": 35.37329, "lon": -119.01871, "name": "Bakersfield, CA"},
    165: {"lat": 29.65163, "lon": -82.32483, "name": "Gainesville, FL"},
    166: {"lat": 40.65010, "lon": -73.94958, "name": "Brooklyn, NY"},
    167: {"lat": 18.46633, "lon": -66.10572, "name": "San Juan, PR"},
    168: {"lat": 33.47097, "lon": -81.97484, "name": "Augusta, GA"},
    169: {"lat": 35.79154, "lon": -78.78112, "name": "Cary, NC"},
    170: {"lat": 33.57786, "lon": -101.85517, "name": "Lubbock, TX"},
    177: {"lat": 41.60054, "lon": -93.60911, "name": "Des Moines, IA"},
    179: {"lat": 33.24866, "lon": -111.63430, "name": "Queen Creek, AZ"},
    189: {"lat": 32.11548, "lon": -81.24706, "name": "Pooler, GA"},
}


def get_coords(loc_number: int) -> dict | None:
    """
    Return {"lat": float, "lon": float, "name": str} for the given Loc Number,
    or None if the location is unknown or its coordinates are still placeholder
    (0.0 / 0.0 -- Gulf of Guinea sentinel).
    """
    entry = LOCATION_COORDS.get(loc_number)
    if entry is None:
        return None
    if entry["lat"] == _PLACEHOLDER_LAT and entry["lon"] == _PLACEHOLDER_LON:
        return None
    return entry
