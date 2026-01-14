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


def get_latest_cv(column_name: str):
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


def append_row(params: dict, c_new: dict):
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_ID)
    worksheet = sheet.worksheet("Sheet1")

    headers = worksheet.row_values(1)
    if not headers:
        raise ValueError("Header row is empty. Put column names in row 1 first.")

    # join param and context into one row
    params.update(c_new)

    # build row aligned to headers
    row = []
    for h in headers:
        v = params.get(h, "")
        # Sheets likes plain Python scalars/strings, thanks chat
        if isinstance(v, (int, float, str)):
            row.append(v)
        else:
            row.append(str(v))

    worksheet.append_row(row, value_input_option="USER_ENTERED")
