# auth.py
import os, msal, requests
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
TENANT_ID = os.getenv("TENANT_ID")
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ["Files.Read"]  # delegated only

SHARED_TOP = "AI for fabrication"                    # exact name in sharedWithMe
SUBPATH_TO_FILE = "Brainstorming for print test/sla_spc_flowrate_channels_ai.xlsx"
SHEET_NAME = "Sheet1"

def get_token():
    app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY)
    flow = app.initiate_device_flow(scopes=SCOPES)
    print(f"Go to {flow['verification_uri']} and enter code {flow['user_code']}")
    return app.acquire_token_by_device_flow(flow)["access_token"]

def main():
    token = get_token()
    H = {"Authorization": f"Bearer {token}"}

    # 1) Find the shared top-level folder by name in "Shared with me"
    swm = requests.get("https://graph.microsoft.com/v1.0/me/drive/sharedWithMe", headers=H)
    swm.raise_for_status()
    entries = swm.json().get("value", [])

    # debug: show available names
    print("sharedWithMe items:", [x.get("name") for x in entries])

    top = next(
        (x for x in entries
         if x.get("remoteItem", {}).get("folder")
         and x.get("name") == SHARED_TOP),
        None
    )
    if not top:
        raise SystemExit(f"Shared folder '{SHARED_TOP}' not found.")

    remote = top["remoteItem"]
    drive_id = remote["parentReference"]["driveId"]   # source drive
    folder_id = remote["id"]                          # shared folder id in that drive
    print("resolved shared folder:", {"drive_id": drive_id, "folder_id": folder_id})

    # 2) Resolve the target file by PATH RELATIVE TO THE FOLDER ID (no children listing)
    rel_path = quote(SUBPATH_TO_FILE, safe="/")
    meta_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{folder_id}:/{rel_path}"
    print("file meta url:", meta_url)

    meta = requests.get(meta_url, headers=H)
    meta.raise_for_status()
    item = meta.json()
    item_id = item["id"]
    print("resolved file:", {"item_id": item_id, "name": item.get("name")})


    # 3) Read usedRange from Sheet1
    used_url = (
        f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item_id}"
        f"/workbook/worksheets/{quote(SHEET_NAME)}/usedRange"
    )
    print("usedRange url:", used_url)

    r = requests.get(used_url, headers=H)
    r.raise_for_status()
    data = r.json()

    print("Used range:", data.get("address"))
    for row in data.get("values", []):
        print(row)

if __name__ == "__main__":
    main()

