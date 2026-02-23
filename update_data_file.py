"""
update_data_file.py
-------------------
Open the Walk-in sales Excel file, add 'Unique Purchased Cards' to Model Data
rows (matched by Loc Number and Date), and save the result to a new sheet
'Data Model Cards'.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "20260202 Walk in sales - v1400.xlsx")
NEW_SHEET_NAME = "Data Model Cards"
CARDS_COL = "Unique Purchased Cards"
CARDS_SHEET = "WI sales"
WI_SALES_HEADER_ROW = 5  # Header is on row 6 (0-indexed: 5)

def main():
    # Read Model Data
    model_df = pd.read_excel(FILE_PATH, sheet_name="Model Data")
    model_df["Date"] = pd.to_datetime(model_df["Date"])

    # Read 'Unique Purchased Cards' from WI sales sheet (header on row 6)
    # Derive Loc Number from first 4 characters of Location column
    cards_df = pd.read_excel(FILE_PATH, sheet_name=CARDS_SHEET, header=WI_SALES_HEADER_ROW)
    cards_df["Loc Number"] = pd.to_numeric(
        cards_df["Location"].astype(str).str[:4],
        errors="coerce",
    )
    cards_df = cards_df.dropna(subset=["Loc Number"])
    cards_df["Loc Number"] = cards_df["Loc Number"].astype(int)
    cards_subset = cards_df[["Loc Number", "Date", CARDS_COL]].copy()
    cards_subset["Date"] = pd.to_datetime(cards_subset["Date"])

    # Merge: add Unique Purchased Cards to Model Data by Loc Number and Date
    result = model_df.merge(
        cards_subset,
        on=["Loc Number", "Date"],
        how="left",
    )

    # Write to new sheet (append to existing workbook)
    with pd.ExcelWriter(FILE_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        result.to_excel(writer, sheet_name=NEW_SHEET_NAME, index=False)

    print(f"Sheet '{NEW_SHEET_NAME}' written with {len(result):,} rows.")


if __name__ == "__main__":
    main()
