# auth.py
import msal, os, json
from dotenv import load_dotenv
from msal import SerializableTokenCache

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
TENANT_ID = os.getenv("TENANT_ID")
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}" # work/school organizations authority
SCOPES = ["Files.ReadWrite"] # delegated permissions 

def get_token() -> str:
    app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY)
    result = app.acquire_token_silent(SCOPES, account=None)
    if not result:
        flow = app.initiate_device_flow(scopes=SCOPES)
        print(flow["message"])
        result = app.acquire_token_by_device_flow(flow)

    return result["access_token"]

