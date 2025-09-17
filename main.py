# main.py
from auth import get_token
from excel_reader import resolve_shared_link, read_used_range

def main():
    token = get_token()
    share_url = "https://csulb-my.sharepoint.com/:x:/r/personal/ava_hedayatipour_csulb_edu/Documents/CSULB/Studnet/AI%20for%20fabrication/William/sla_spc_flowrate_channels_13batches.xlsx?d=wed55877a245845d9b949a46f573d5120&csf=1&web=1&e=f9SBCm"
    drive_id, item_id = resolve_shared_link(token, share_url)
    df = read_used_range(token, drive_id, item_id, sheet="Sheet1")
    print(df.head())

if __name__ == "__main__":
    main()



