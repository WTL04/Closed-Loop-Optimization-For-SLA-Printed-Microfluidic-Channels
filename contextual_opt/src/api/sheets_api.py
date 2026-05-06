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
    "batch_id",
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


def pullData(
    sheet_name: str = "Reformated - Experiment Realistic Deltas", verbose: bool = True
):
    """
    Pulls data from google sheets from the cloud

    Args:
        verbose: bool
            print the dataframe results
    Returns:
        pandas DataFrame from data in google sheets
    """
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


def get_latest_col_value(
    column_name: str,
    sheet_name: str = "Sheet1",
):
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


def append_row(
    batch_id: int,
    params: dict,
    c_new: dict,
    sheet_name: str = "Sheet1",
):
    """
    Append a single experiment record to Google Sheets.

    - batch_id is metadata (stored as string)
    - params and context are numeric features where possible
    - Row is aligned strictly to existing sheet headers
    """
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_ID)
    worksheet = sheet.worksheet(sheet_name)

    headers = worksheet.row_values(1)
    if not headers:
        raise ValueError("Header row is empty. Put column names in row 1 first.")

    # metadata (identifiers)
    metadata = {
        "batch_id": str(batch_id),
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

    worksheet.append_row(row, value_input_option="USER_ENTERED")


def update_row(batch_id: int, updates: dict, sheet_name: str = "Sheet1"):
    """
    Update specific columns in an existing row identified by batch_id.

    Args:
        batch_id: The batch_id to look up in column 1
        updates: Dict of {column_name: value} to write into that row
        sheet_name: Worksheet name
    """

    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_ID)
    worksheet = sheet.worksheet(sheet_name)

    headers = worksheet.row_values(1)
    if not headers:
        raise ValueError("Header row is empty.")

    # Find the row index by batch_id (column 1)
    batch_col = worksheet.col_values(1)
    try:
        row_index = batch_col.index(str(batch_id)) + 1  # 1-indexed
    except ValueError:
        raise ValueError(f"batch_id {batch_id} not found in sheet '{sheet_name}'")

    # Write only the specified columns
    for col_name, value in updates.items():
        if col_name not in headers:
            continue
        col_index = headers.index(col_name) + 1  # 1-indexed
        try:
            worksheet.update_cell(row_index, col_index, float(value))
        except (TypeError, ValueError):
            worksheet.update_cell(row_index, col_index, value)
