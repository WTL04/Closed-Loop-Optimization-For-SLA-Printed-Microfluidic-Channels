# excel_reader.py
import base64, requests, pandas as pd
import urllib.parse


def resolve_shared_link(token: str, share_url: str):
    """Resolve a OneDrive/SharePoint share URL into driveId + itemId."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Prefer": "redeemSharingLink"
    }

    # Graph needs the share URL base64-encoded in a "u!" format
    b64 = base64.urlsafe_b64encode(share_url.encode("utf-8")).decode("utf-8").rstrip("=")
    r = requests.get(f"https://graph.microsoft.com/v1.0/shares/u!{b64}/driveItem", headers=headers)
    r.raise_for_status()
    item = r.json()

    return item["parentReference"]["driveId"], item["id"]

def read_used_range(token: str, drive_id: str, item_id: str, sheet="Sheet1") -> pd.DataFrame:
    """Read all used cells from a worksheet into a DataFrame."""
    headers = {"Authorization": f"Bearer {token}"}

    # start a workbook session
    sess = requests.post(
        f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item_id}/workbook/createSession",
        headers=headers, json={"persistChanges": True}
    ).json()
    sid = sess["id"]
    wh = {**headers, "Workbook-Session-Id": sid}

    # get used range
    res = requests.get(
        f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item_id}"
        f"/workbook/worksheets('{sheet}')/usedRange(valuesOnly=true)",
        headers=wh
    ).json()
    values = res["values"]

    # close session
    requests.post(
        f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item_id}/workbook/closeSession",
        headers=wh
    )

    # convert to DataFrame (assumes first row is headers)
    return pd.DataFrame(values[1:], columns=values[0])
