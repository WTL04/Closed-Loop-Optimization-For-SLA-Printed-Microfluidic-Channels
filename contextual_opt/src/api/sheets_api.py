import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from dotenv import load_dotenv
import os

# load environmental variables
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


# Expected headers matching the dataset schema
EXPECTED_HEADERS = [
    "channel",
    "layer_thickness_um",
    "ambient_temp",
    "resin_temp",
    "resin_age",
    "channel_length_um",
    "channel_width_um",
    "channel_height_um",
    "delta_length_um",
    "delta_width_um",
    "delta_height_um",
    "dim_error",
    "flow_rate",
]


def pullData(sheet_name: str = "Experiment Realistic Deltas", verbose: bool = True):
    """
    Pulls data from google sheets from the cloud

    Args:
        verbose: bool
            print the dataframe results
    Returns:
        pandas DataFrame from data in google sheets
    """
    try:
        # authorize client with credentials
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID)
        worksheet = sheet.worksheet(sheet_name)

        # Use expected_headers to handle duplicate/empty headers in sheet
        data = worksheet.get_all_records(expected_headers=EXPECTED_HEADERS)
        df = pd.DataFrame(data)
        if verbose:
            print(df)

        return df
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"ERROR: Spreadsheet not found. SHEET_ID={SHEET_ID}")
        raise
    except gspread.exceptions.WorksheetNotFound:
        print(f"ERROR: Worksheet '{sheet_name}' not found")
        raise
    except Exception as e:
        print(f"ERROR: Failed to pull data from Google Sheets: {e}")
        raise


def get_latest_col_value(
    column_name: str,
    sheet_name: str = "Sheet1",
):
    try:
        # authorize client with credentials
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID)
        worksheet = sheet.worksheet(sheet_name)

        # fetch header row
        headers = worksheet.row_values(1)
        if column_name not in headers:
            raise ValueError(f"Column '{column_name}' not found")

        col_idx = headers.index(column_name) + 1  # 1-based indexing

        # get all column values
        col_values = worksheet.col_values(col_idx)[1:]

        # filter empty cells
        col_values = [v for v in col_values if v != ""]

        if not col_values:
            return None

        # return last value, aka most recent recorded cv
        return col_values[-1]
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"ERROR: Spreadsheet not found. SHEET_ID={SHEET_ID}")
        raise
    except gspread.exceptions.WorksheetNotFound:
        print(f"ERROR: Worksheet '{sheet_name}' not found")
        raise
    except Exception as e:
        print(f"ERROR: Failed to get latest column value: {e}")
        raise


def append_row(
    channel: int,
    params: dict,
    c_new: dict,
    sheet_name: str = "Sheet1",
):
    """
    Append a single experiment record to Google Sheets.

    - channel is metadata (stored as string)
    - params and context are numeric features where possible
    - Row is aligned strictly to existing sheet headers
    """
    try:
        # authorize client with credentials
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID)
        worksheet = sheet.worksheet(sheet_name)

        headers = worksheet.row_values(1)
        if not headers:
            raise ValueError("Header row is empty. Put column names in row 1 first.")

        # metadata (identifiers)
        metadata = {
            "channel": str(channel),
        }

        # numeric features (params + context)
        features = {}
        for k, v in {**params, **c_new}.items():
            try:
                features[k] = float(v)
            except (TypeError, ValueError):
                features[k] = ""

        # build row strictly following header order
        row = []
        for h in headers:
            if h in metadata:
                row.append(metadata[h])
            elif h in features:
                row.append(features[h])
            else:
                row.append("")

        # Find the last row with data in column 1 (channel column)
        all_values = worksheet.get_all_values()
        last_row_with_data = 0
        for i, row_vals in enumerate(all_values):
            if row_vals and row_vals[0]:
                last_row_with_data = i + 1

        next_row = last_row_with_data + 1

        # Write to the correct row and columns
        for col_idx, value in enumerate(row, start=1):
            worksheet.update_cell(next_row, col_idx, value)

        print(f"SUCCESS: Appended channel {channel} to '{sheet_name}'")
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"ERROR: Spreadsheet not found. SHEET_ID={SHEET_ID}")
        raise
    except gspread.exceptions.WorksheetNotFound:
        print(f"ERROR: Worksheet '{sheet_name}' not found")
        raise
    except Exception as e:
        print(f"ERROR: Failed to append row to Google Sheets: {e}")
        raise


def update_row(channel: int, updates: dict, sheet_name: str = "Sheet1"):
    """
    Update specific columns in an existing row identified by channel.

    Args:
        channel: The channel to look up in column 1
        updates: Dict of {column_name: value} to write into that row
        sheet_name: Worksheet name
    """
    try:
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID)
        worksheet = sheet.worksheet(sheet_name)

        headers = worksheet.row_values(1)
        if not headers:
            raise ValueError("Header row is empty.")

        # Find the row index by channel (column 1)
        channel_col = worksheet.col_values(1)
        try:
            row_index = channel_col.index(str(channel)) + 1  # 1-indexed
        except ValueError:
            raise ValueError(f"channel {channel} not found in sheet '{sheet_name}'")

        # Write only the specified columns
        for col_name, value in updates.items():
            if col_name not in headers:
                continue
            col_index = headers.index(col_name) + 1  # 1-based indexing
            try:
                worksheet.update_cell(row_index, col_index, float(value))
            except (TypeError, ValueError):
                worksheet.update_cell(row_index, col_index, value)
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"ERROR: Spreadsheet not found. SHEET_ID={SHEET_ID}")
        raise
    except gspread.exceptions.WorksheetNotFound:
        print(f"ERROR: Worksheet '{sheet_name}' not found")
        raise
    except Exception as e:
        print(f"ERROR: Failed to update row in Google Sheets: {e}")
        raise
