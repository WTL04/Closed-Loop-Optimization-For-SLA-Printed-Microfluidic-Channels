# auth.py
import msal, os, requests, urllib
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
TENANT_ID = os.getenv("TENANT_ID")
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ["Files.Read"]

def get_token():
    app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY)
    flow = app.initiate_device_flow(scopes=SCOPES)
    print(f"Go to {flow['verification_uri']} and enter code {flow['user_code']}")
    result = app.acquire_token_by_device_flow(flow)
    return result["access_token"]

def main():
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}

    file_path = "/Documents/CSULB/Studnet/AI for fabrication/Brainstorming for print test/sla_spc_flowrate_channels_ai.xlsx"

    # Resolve file -> item_id
    enc_path = urllib.parse.quote(file_path)  # encode spaces etc.
    meta_url = f"https://graph.microsoft.com/v1.0/me/drive/root:{enc_path}"
    meta = requests.get(meta_url, headers=headers)
    meta.raise_for_status()
    item_id = meta.json()["id"]


    # Encode sheet name
    sheet_name = "Sheet1"
    enc_sheet = urllib.parse.quote(sheet_name)

   # Read usedRange (i.e., A1 to the bottom-right of used cells)
    used_url = f"https://graph.microsoft.com/v1.0/me/drive/items/{item_id}/workbook/worksheets/{enc_sheet}/usedRange"
    r = requests.get(used_url, headers=headers)
    r.raise_for_status()
    data = r.json()

    # print rows
    print("Used range:", data.get("address"))
    for row in data.get("values", []):
        print(row)


if __name__ == "__main__":
    main()
