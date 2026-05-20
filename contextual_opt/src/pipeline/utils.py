"""
Utility functions for CBO pipeline.

Functions:
- print_suggested_params: Debug print suggested parameters
- save_params_to_json: Save parameters to JSON file
- append_single_to_sheets: Append results to Google Sheets with retry
"""

import json
import time

from contextual_opt.src.api.sheets_api import get_latest_col_value, append_row


def print_suggested_params(suggested_params: dict):
    """Print suggested parameters in readable format."""
    print("\n=== Suggested Parameters ===")
    print(f"Length: {suggested_params.get('channel_length_um', 'N/A'):.1f} µm")
    print(f"Width: {suggested_params.get('channel_width_um', 'N/A'):.1f} µm")
    print(f"Height: {suggested_params.get('channel_height_um', 'N/A'):.1f} µm")
    print(f"Layer Thickness: {suggested_params.get('layer_thickness_um', 'N/A')} µm")


def save_params_to_json(suggested_params: dict, channel: int):
    """Save suggested parameters to JSON file."""
    filename = "contextual_opt/src/data/suggested_params.json"
    data = {"channel": channel, **suggested_params}
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def append_single_to_sheets(
    channel_results: dict, context: dict, sheet_name: str, max_retries: int = 5
):
    """
    Append one row to Google Sheets with exponential backoff retry.

    Args:
        channel_results: Dict with channel data (channel_length_um, dim_error, flow_rate, etc.)
        context: Dict with context parameters (layer_thickness_um, ambient_temp, etc.)
        sheet_name: Name of the Google Sheet tab
        max_retries: Number of retries on network failure (default 5)
    """
    retries = 0
    backoff = 2  # 2, 4, 8, 16, 32 seconds
    channel = None
    row_data = None

    while True:
        try:
            channel_raw = get_latest_col_value(column_name="channel", sheet_name=sheet_name)
            channel = int(channel_raw) + 1 if channel_raw is not None else 1

            row_data = {
                "channel": channel,
                "layer_thickness_um": context.get("layer_thickness_um", 100),
                "ambient_temp": context.get("ambient_temp", 80.0),
                "resin_temp": context.get("resin_temp", 80.0),
                "resin_age": context.get("resin_age", 15.0),
                "channel_length_um": channel_results["channel_length_um"],
                "channel_width_um": channel_results["channel_width_um"],
                "channel_height_um": channel_results["channel_height_um"],
                "delta_length_um": channel_results["delta_length_um"],
                "delta_width_um": channel_results["delta_width_um"],
                "delta_height_um": channel_results["delta_height_um"],
                "dim_error": channel_results["dim_error"],
                "flow_rate": channel_results["flow_rate"],
            }

            append_row(channel, row_data, context, sheet_name=sheet_name)
            print(f"Appended channel {row_data['channel']} to '{sheet_name}'")
            return

        except Exception as e:
            retries += 1
            if retries > max_retries:
                print(f"[ERROR] Max retries ({max_retries}) reached. Could not sync with Google Sheets.")
                fallback_file = f"contextual_opt/src/data/fallback_channel_{channel}.json"
                with open(fallback_file, "w") as f:
                    json.dump(row_data, f, indent=2)
                print(f"Saved result to local fallback: {fallback_file}")
                raise

            wait = backoff ** retries
            print(f"[WARNING] Network error pushing to Sheets: {e}")
            print(f"Retrying in {wait}s... (attempt {retries}/{max_retries})")
            time.sleep(wait)