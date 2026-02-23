"""
weather.py
----------
Open-Meteo API client for daily precipitation data.

Public API
----------
get_precipitation(lat, lon, date) -> float | None
    Returns the daily precipitation sum in INCHES for the given coordinates
    and date. Returns None on any failure (network error, timeout, date out
    of supported range, unexpected API response, etc.).

Endpoints used
--------------
- Forecast  : https://api.open-meteo.com/v1/forecast
              Used for today through today + 16 days.
- Archive   : https://archive-api.open-meteo.com/v1/archive
              Used for any date in the past.

No API key required for non-commercial use.

Caching
-------
Results are cached for 1 hour via @st.cache_data(ttl=3600).
Each unique (lat, lon, date) triple is fetched at most once per hour,
so uploading a file with repeated (location, date) rows only hits the
API once per unique pair.
"""

import datetime
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FORECAST_URL        = "https://api.open-meteo.com/v1/forecast"
_ARCHIVE_URL         = "https://archive-api.open-meteo.com/v1/archive"
_TIMEOUT_SEC         = 10
_FORECAST_HORIZON    = 16   # days Open-Meteo forecasts ahead


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _is_forecast_date(date: datetime.date) -> bool:
    """True if date falls within the Open-Meteo forecast window."""
    today = datetime.date.today()
    return today <= date <= today + datetime.timedelta(days=_FORECAST_HORIZON)


def _fetch_precipitation(lat: float, lon: float, date_str: str) -> float | None:
    """
    Internal HTTP call — not cached, not called directly by the UI.
    Selects the forecast or archive endpoint based on the date.
    Returns precipitation in inches, or None on any failure.
    date_str must be ISO format: "YYYY-MM-DD".
    """
    date = datetime.date.fromisoformat(date_str)
    url  = _FORECAST_URL if _is_forecast_date(date) else _ARCHIVE_URL

    params = {
        "latitude":           lat,
        "longitude":          lon,
        "daily":              "precipitation_sum",
        "precipitation_unit": "inch",
        "timezone":           "auto",
        "start_date":         date_str,
        "end_date":           date_str,
    }

    try:
        resp = requests.get(url, params=params, timeout=_TIMEOUT_SEC)
        resp.raise_for_status()
        data = resp.json()

        times  = data.get("daily", {}).get("time", [])
        values = data.get("daily", {}).get("precipitation_sum", [])

        if not times or not values:
            return None

        if times[0] == date_str and values[0] is not None:
            return round(float(values[0]), 4)

        return None

    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.HTTPError:
        return None
    except (KeyError, IndexError, ValueError, TypeError):
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_precipitation(lat: float, lon: float, date: datetime.date) -> float | None:
    """
    Fetch daily precipitation (in inches) for (lat, lon) on the given date.

    - For dates within the next 16 days: uses the Open-Meteo forecast API.
    - For past dates: uses the Open-Meteo historical archive API.

    Returns None if the fetch fails for any reason. Always check for None
    before using the result.

    Results are cached for 1 hour. Repeated calls with the same arguments
    return the cached value instantly without hitting the network.
    """
    return _fetch_precipitation(lat, lon, date.isoformat())
