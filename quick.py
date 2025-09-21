# quick_probe_workbook.py
import os, msal, requests
from dotenv import load_dotenv
from urllib.parse import quote

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
TENANT_ID = os.getenv("TENANT_ID")
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ["Files.Read"]

# from your shared_quick_check output (idx 0 = the XLSX)
DRIVE_ID = "b!5XP7pdFyG0m7pl7KO2k9f_LlDmnM0FNOmWuydHoAafN1C7xNnKS1SLhHipqfVHI1"
ITEM_ID  = "01GMGSNCL2Q5K62WBE3FC3SSNEN5LT2UJA"
SHEET    = "Sheet1"

def token():
    app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY)
    flow = app.initiate_device_flow(scopes=SCOPES)
    print(f"Go to {flow['verification_uri']} and enter code {flow['user_code']}")
    return app.acquire_token_by_device_flow(flow)["access_token"]

T = token()
H = {"Authorization": f"Bearer {T}"}

# 1) List worksheets to confirm names
ws = requests.get(
    f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/items/{ITEM_ID}/workbook/worksheets",
    headers=H,
)
print("worksheets status:", ws.status_code)
print("worksheets:", [w["name"] for w in ws.json().get("value", [])])

# 2) Read used range from Sheet1 (NOTE the trailing /usedRange)
ur = requests.get(
    f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/items/{ITEM_ID}"
    f"/workbook/worksheets/{quote(SHEET)}/usedRange",
    headers=H,
)
print("usedRange status:", ur.status_code)
ur.raise_for_status()
data = ur.json()
print("Used range:", data.get("address"))
for row in data.get("values", []):
    print(row)

