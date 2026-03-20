import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Use the Archive API so results match the documented example URL exactly.
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": [33.749, 39.95238, 26.0112, 34.06334, 39.12711, 39.73915, 43.1009, 33.66946, 40.84816, 33.78779, 40.0334, 29.42412, 38.62727, 30.26715, 30.33218, 41.82399, 37.33939, 39.83665, 40.44062, 32.71571, 25.77427, 33.15067, 41.4995, 40.80426, 43.70643, 34.13973, 39.19344, 35.40888, 35.92506, 29.76328, 36.16589, 33.50921, 40.75566, 33.95621, 41.25626, 39.09973, 40.71427, 45.07246, 33.41477, 40.10233, 32.73569, 37.55376, 36.15398, 39.76838, 39.96118, 43.04946, 38.75212, 42.25288, 32.78306, 45.40762, 28.53834, 35.46756, 41.63031, 43.6135, 36.85293, 42.65258, 43.04812, 34.85262, 42.36837, 33.9603, 42.21947, 30.15946, 34.05223, 35.08449, 42.99564, 32.83707, 40.90982, 29.42412, 42.47926, 42.86947, 42.88645, 44.88969, 36.74773, 29.5294, 33.53865, 38.78953, 31.75872, 43.15478, 36.17497, 38.88511, 34.19543, 34.74648, 43.45011, 38.99067, 41.66394, 38.98223, 37.70577, 33.15809, 34.00071, 32.22174, 29.95465, 33.68906, 34.07538, 26.20341, 61.21806, 35.0832, 34.22834, 18.38079, 40.92538, 47.30732, 39.29038, 36.33202, 40.5576, 35.14953, 27.94752, 43.07305, 33.83585, 40.76078, 41.22232, 41.99531, 34.17056, 33.52066, 38.84622, 40.56233, 38.25424, 40.2737, 27.80058, 40.1138, 29.21081, 26.62168, 35.86815, 36.09986, 33.44734, 39.14344, 30.18022, 42.28343, 34.7304, 37.69224, 40.79895, 42.99564, 37.6391, 47.61038, 39.89281, 38.24936, 33.76696, 39.61366, 37.97798, 47.82093, 35.04563, 40.60843, 44.51916, 43.54997, 40.6501, 35.37329, 29.65163, 40.6501, 18.46633, 33.47097, 35.79154, 33.57786, 41.60054, 33.24866, 21.3099],
	"longitude": [-84.38798, -75.16362, -80.14949, -117.65089, -84.51439, -104.9847, -75.23266, -117.82311, -73.99764, -117.85311, -83.15825, -98.49363, -90.19789, -97.74306, -81.65565, -71.41283, -121.89496, -105.0372, -79.99589, -117.16472, -80.19366, -96.82361, -81.69541, -73.169, -79.39864, -118.03534, -76.72442, -80.58158, -86.86889, -95.36327, -86.78444, -111.89903, -73.58763, -83.98796, -95.94043, -94.57857, -74.00597, -93.45579, -111.90931, -75.27435, -97.10807, -77.46026, -95.99277, -86.15804, -82.99879, -88.00759, -121.28801, -71.00227, -96.80667, -122.57037, -81.37924, -97.51643, -87.85394, -116.20345, -75.97799, -73.75623, -76.14742, -82.39401, -83.35271, -118.4309, -87.97952, -85.65983, -118.24368, -106.65114, -71.45479, -97.08195, -73.80791, -98.49363, -71.15228, -85.64475, -78.87837, -93.34995, -119.77237, -95.20104, -112.18599, -77.1872, -106.48693, -77.61556, -115.13722, -76.91581, -79.76256, -92.28959, -79.68292, -77.02609, -83.55521, -94.67079, -122.46192, -117.35059, -81.03481, -110.92648, -90.07507, -78.88669, -84.29409, -98.23001, -149.90028, -80.8923, -118.53675, -66.15328, -74.27654, -122.22845, -76.61219, -94.11854, -74.28459, -90.04898, -82.45843, -89.40123, -118.34063, -111.89105, -73.0565, -87.88451, -118.83759, -86.80249, -77.30637, -74.13986, -85.75941, -76.88442, -97.39638, -75.071, -81.02283, -81.84059, -83.56184, -80.24422, -84.14686, -77.20137, -95.45577, -71.3495, -86.58594, -97.33754, -81.37845, -71.45479, -120.99688, -122.20068, -75.12041, -122.03997, -118.18923, -86.10665, -122.03107, -122.31513, -85.30968, -75.49018, -88.01983, -96.70033, -73.94958, -119.01871, -82.32483, -73.94958, -66.10572, -81.97484, -78.78112, -101.85517, -93.60911, -111.6343, 157.8581],
	"start_date": "2026-02-04",
	"end_date": "2026-03-05",
	"daily": "rain_sum",
	"precipitation_unit": "inch",
}
responses = openmeteo.weather_api(url, params=params)

# Read location2: meteo-index (1-based) -> ID for CSV output
location2_path = "location2.csv"
loc2 = pd.read_csv(location2_path)
meteo_index_to_id = loc2.set_index("meteo-index")["ID"].to_dict()

# Collect all daily data into rows: meteo-index, date (YYYY-MM-DD), precipitation
all_rows = []
for meteo_index, response in enumerate(responses, start=1):
	# Process daily data. The order of variables needs to be the same as requested.
	daily = response.Daily()
	daily_precipitation = daily.Variables(0).ValuesAsNumpy()
	
	dates = pd.date_range(
		start=pd.to_datetime(daily.Time(), unit="s", utc=True),
		end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
		freq=pd.Timedelta(seconds=daily.Interval()),
		inclusive="left"
	)
	
	for date, precip in zip(dates, daily_precipitation):
		all_rows.append({
			"Loc Number": meteo_index,
			"Date": date.strftime("%Y-%m-%d"),
			"Precipitation": float(precip),
		})

df = pd.DataFrame(all_rows)
# Convert Loc Number from meteo-index to location ID from location2
df["Loc Number"] = df["Loc Number"].map(meteo_index_to_id)
output_path = "historical_precipitation_20260308.csv"
# Print a sample of the API-derived data before writing to file
#print("Preview of API response data (first 10 rows):")
#print(df.head(25))
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} rows to {output_path}")