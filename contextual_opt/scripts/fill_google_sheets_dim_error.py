#!/usr/bin/env python3
"""
Script to fill empty dim_error values in Google Sheets.

Reads data from a Google Sheet tab, computes dim_error for rows
where dim_error is empty, and updates the sheet with computed values.

Formula:
    fabricated = target - delta
    dim_error = NMSE(fabricated vs nominal) in um^2

Nominal dimensions: 40000 x 500 x 500 um
"""

import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()


SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
SHEET_ID = os.getenv("SHEET_ID")
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

creds = Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=SCOPES,
)

NOMINAL_DIMENSIONS = {
    "length": 40000,
    "width": 500,
    "height": 500,
}


def compute_dimensional_error(params: dict) -> float:
    """
    Compute Normalized Mean Squared Error (NMSE)
    between the fabricated dimensions (Ax suggested) and the
    nominal target dimensions (40000 x 500 x 500 µm).

    Uses exact column names from dataset:
    - channel_length_um, channel_width_um, channel_height_um (µm)
    - delta_length_um, delta_width_um, delta_height_um (µm)

    Args:
        params: Dict with channel_length_um/channel_width_um/channel_height_um (µm) and delta_*_um (µm)
        num_channels: Number of channels (default: NUM_CHANNELS from config)

    Returns:
        Normalized mean squared error in µm^2 (lower is better)
    """

    nominal_length = NOMINAL_DIMENSIONS["length"]
    nominal_width = NOMINAL_DIMENSIONS["width"]
    nominal_height = NOMINAL_DIMENSIONS["height"]

    # get exact columns from dataset
    length = params.get("channel_length_um", nominal_length)
    width = params.get("channel_width_um", nominal_width)
    height = params.get("channel_height_um", nominal_height)
    delta_length = params.get("delta_length_um", 0.0) or 0.0
    delta_width = params.get("delta_width_um", 0.0) or 0.0
    delta_height = params.get("delta_height_um", 0.0) or 0.0

    try:
        # deltas are in µm
        length_delta = float(delta_length)
        width_delta = float(delta_width)
        height_delta = float(delta_height)
    except (TypeError, ValueError):
        length_delta = 0.0
        width_delta = 0.0
        height_delta = 0.0

    # fabricated dimensions = ax suggested - random deltas
    fabricated_length = float(length) - length_delta
    fabricated_width = float(width) - width_delta
    fabricated_height = float(height) - height_delta

    # calculate NMSE
    squared_errors = [
        ((fabricated_length - nominal_length) / nominal_length) ** 2,
        ((fabricated_width - nominal_width) / nominal_width) ** 2,
        ((fabricated_height - nominal_height) / nominal_height) ** 2,
    ]

    return float(np.mean(squared_errors)) if squared_errors else 0.0


def fill_google_sheet_dim_error(sheet_name: str = None):
    """Pull data, compute dim_error, update Google Sheet."""

    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_ID)

    if sheet_name is None:
        print("Available worksheets:")
        for i, ws in enumerate(sheet.worksheets()):
            print(f"  {i + 1}) {ws.title}")
        sheet_name = input("\nEnter worksheet name: ").strip()

    worksheet = sheet.worksheet(sheet_name)

    print(f"\nFetching data from '{sheet_name}'...")
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)

    print(f"Total rows: {len(df)}")

    # Check if dim_error column exists
    if "dim_error" not in df.columns:
        print("ERROR: 'dim_error' column not found in sheet")
        print(f"Available columns: {list(df.columns)}")
        return

    # Find rows with empty dim_error
    # Handle both empty string and NaN
    empty_mask = df["dim_error"].isna() | (df["dim_error"] == "")
    rows_to_fill = df[empty_mask].index.tolist()

    print(f"Rows with empty dim_error: {len(rows_to_fill)}")

    if len(rows_to_fill) == 0:
        print("No empty dim_error values found.")
        return

    # Compute dim_error for each empty row and store
    headers = worksheet.row_values(1)
    dim_error_col_idx = headers.index("dim_error") + 1  # 1-based

    updates = []  # Collect all updates as (row, col, value) tuples

    for idx in rows_to_fill:
        row_num = idx + 2  # +2: +1 for header, +1 for 1-based index

        row_data = df.iloc[idx]

        # Build params dict for compute_dimensional_error
        params = {
            "channel_length_um": row_data.get("channel_length_um"),
            "channel_width_um": row_data.get("channel_width_um"),
            "channel_height_um": row_data.get("channel_height_um"),
            "delta_length_um": row_data.get("delta_length_um"),
            "delta_width_um": row_data.get("delta_width_um"),
            "delta_height_um": row_data.get("delta_height_um"),
        }

        dim_error = compute_dimensional_error(params)

        # Store as (row, col, value)
        updates.append((row_num, dim_error_col_idx, dim_error))

    # Batch write all at once using update_cells
    # Format: rows, cols, values as list of lists
    if updates:
        # Sort by row number
        updates.sort(key=lambda x: x[0])

        # Get unique rows and prepare data
        row_to_values = {}
        for row, col, value in updates:
            if row not in row_to_values:
                row_to_values[row] = {}
            row_to_values[row][col] = value

        # Update in batches of 50 rows to avoid quota issues
        batch_size = 50
        rows = sorted(row_to_values.keys())
        total_rows = len(rows)

        for batch_start in range(0, total_rows, batch_size):
            batch_rows = rows[batch_start : batch_start + batch_size]

            # Build cell list for batch
            cell_list = []
            for row in batch_rows:
                # Get values for all columns in this row that need updating
                for col in sorted(row_to_values[row].keys()):
                    cell = worksheet.cell(row, col)
                    cell.value = row_to_values[row][col]
                    cell_list.append(cell)

            if cell_list:
                worksheet.update_cells(cell_list)
                print(
                    f"  Batch {batch_start // batch_size + 1}: Updated rows {batch_start + 1}-{min(batch_start + batch_size, total_rows)}"
                )

    print(f"\nCompleted! Updated {len(updates)} rows.")

    # Show sample of results
    print("\nSample of computed dim_error values:")
    sample_df = df.iloc[rows_to_fill[:5]][
        ["batch_id", "channel", "delta_length_um"]
    ].copy()
    sample_df["dim_error"] = [
        compute_dimensional_error(
            {
                "channel_length_um": df.iloc[i]["channel_length_um"],
                "channel_width_um": df.iloc[i]["channel_width_um"],
                "channel_height_um": df.iloc[i]["channel_height_um"],
                "delta_length_um": df.iloc[i]["delta_length_um"],
                "delta_width_um": df.iloc[i]["delta_width_um"],
                "delta_height_um": df.iloc[i]["delta_height_um"],
            }
        )
        for i in rows_to_fill[:5]
    ]
    print(sample_df.to_string(index=False))


if __name__ == "__main__":
    import sys

    sheet_name = sys.argv[1] if len(sys.argv) > 1 else None
    fill_google_sheet_dim_error(sheet_name)
