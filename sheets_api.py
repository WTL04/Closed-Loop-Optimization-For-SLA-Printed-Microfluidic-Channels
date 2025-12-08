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
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
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
