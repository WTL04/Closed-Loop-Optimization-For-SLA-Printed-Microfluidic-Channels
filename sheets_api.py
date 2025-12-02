import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
