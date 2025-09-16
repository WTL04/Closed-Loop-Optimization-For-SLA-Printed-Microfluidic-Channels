# auth.py  (DELEGATED)
import os, msal
from dotenv import load_dotenv

load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")  # App registrations → Application (client) ID
TENANT_ID = os.getenv("TENANT_ID")  # Entra ID → Overview → Tenant ID (GUID)
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"   # if app is single-tenant
SCOPES = ["User.Read", "Files.ReadWrite", "Sites.Read.All"]    # delegated scopes

app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY)

result = app.acquire_token_silent(SCOPES, account=None)
if not result:
    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        raise RuntimeError(f"Device flow failed to start: {flow}")
    print(flow["message"])  # follow the URL & code shown
    result = app.acquire_token_by_device_flow(flow)

access_token = result["access_token"]
print("Delegated token:", access_token[:40], "…")

