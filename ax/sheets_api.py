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


def pullData(verbose=True):
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
    worksheet = sheet.worksheet("Sheet1")

    data = worksheet.get_all_records()  # list of dicts
    df = pd.DataFrame(data)
    if verbose:
        print(df)

    return df


def get_latest_col_value(column_name: str):
    # authorize client with credentials
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_ID)
    worksheet = sheet.worksheet("Sheet1")

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


def append_row(batch_id: int, num_channels: int, params: dict, c_new: dict):
    """
    Append a single experiment record to Google Sheets.

    - batch_id and channel_id are metadata (stored as strings)
    - params and context are numeric features where possible
    - Row is aligned strictly to existing sheet headers
    """
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_ID)
    worksheet = sheet.worksheet("Sheet1")

    headers = worksheet.row_values(1)
    if not headers:
        raise ValueError("Header row is empty. Put column names in row 1 first.")

    for channel_id in range(num_channels - 1):
        # metadata (identifiers)
        metadata = {
            "batch_id": str(batch_id),
            "channel_id": channel_id + 1,
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
