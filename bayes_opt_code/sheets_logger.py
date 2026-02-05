import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

class GoogleSheetsLogger:
    def __init__(self, creds_path, sheet_id, worksheet_name):
        creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(sheet_id)
        self.ws = sheet.worksheet(worksheet_name)

    def append_dataframe(self, df):
        header = self.ws.row_values(1)
        if not header:
            raise RuntimeError("Worksheet header row is empty.")

        for col in header:
            if col not in df.columns:
                df[col] = ""

        df_out = df[header]
        rows = df_out.astype(object).where(df_out.notnull(), "").values.tolist()
        self.ws.append_rows(rows, value_input_option="USER_ENTERED")

    def read_dataframe(self):
        """
        Reads the active worksheet into a pandas DataFrame. 
        Numeric columns are converted when possible. 
        Blank flow cells are stored as NaN.
        """
        import pandas as pd

        values = self.ws.get_all_values()
        if not values or len(values) < 2:
            return pd.DataFrame()

        header = values[0]
        rows = values[1:]
        df = pd.DataFrame(rows, columns=header)

        # numeric conversions
        numeric_cols = [
            "batch_id",
            "channel_id",
            "layer_thickness_um",
            "z_rotation_deg",
            "fit_adjustment",
            "resin_age",
            "resin_temp",
            "ambient_temp",
            "channel_flow_rate_ml_per_min",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df