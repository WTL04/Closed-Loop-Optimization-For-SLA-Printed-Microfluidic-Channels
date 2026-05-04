"""
Data loading functions for CSV and Google Sheets.

Functions:
- load_dataset
- load_data_source
- extract_channel_data
"""

import pandas as pd

from contextual_opt.src.pipeline.config import NOMINAL_DIMENSIONS
from contextual_opt.src.api.sheets_api import pullData
from contextual_opt.src.pipeline.computation import compute_dimensional_error


def load_dataset(
    is_testing: bool, sheet_name: str = "Experiment Random Deltas", verbose=True
):
    """
    Returns DataFrame from Google Spreadsheet or a chosen fake dataset.
    Long format: one row per channel.

    Columns: | batch | channel | channel_length | channel_width | channel_height | delta_length | ... | flow_rate |

    Args:
        is_testing: bool
            Uses fake dataset when True, Google Spreadsheet when False
        sheet_name: str
            Name of the Google Sheets tab (default: "Experiment Random Deltas")
    """
    if is_testing:
        print("\n1) experiment_realistic_deltas.csv \n2) experiment_random_deltas.csv")
        choice = input("\nPlease choose one of the two: ")

        if choice == "1":
            path = "contextual_opt/datasets/experiment_realistic_deltas.csv"
        elif choice == "2":
            path = "contextual_opt/datasets/experiment_random_deltas.csv"
        else:
            raise ValueError("Invalid fake dataset option")

        if verbose:
            print(f"Loading fake dataset: {path}")
        df = pd.read_csv(path)
    else:
        df = pullData(sheet_name=sheet_name, verbose=verbose)

    # If no 'channel' column, compute dimensional_error using exact column names
    if "channel" not in df.columns:
        df["dim_error"] = df.apply(
            lambda row: compute_dimensional_error(
                {
                    "channel_length_mm": row.get(
                        "channel_length_mm", NOMINAL_DIMENSIONS["length"]
                    ),
                    "channel_width_mm": row.get(
                        "channel_width_mm", NOMINAL_DIMENSIONS["width"]
                    ),
                    "channel_height_mm": row.get(
                        "channel_height_mm", NOMINAL_DIMENSIONS["height"]
                    ),
                    "delta_length_um": row.get("delta_length_um", 0.0) or 0.0,
                    "delta_width_um": row.get("delta_width_um", 0.0) or 0.0,
                    "delta_height_um": row.get("delta_height_um", 0.0) or 0.0,
                },
                num_channels=1,
            ),
            axis=1,
        )

    if verbose:
        print(f"Loaded {len(df)} rows (channels) from {sheet_name}")

    return df


def load_data_source(
    sheet_name: str = "Experiment Random Deltas", is_testing: bool = False
):
    """
    Load data from Google Sheets or fake testing data.

    Args:
        sheet_name: str
            Name of the Google Sheets tab
        is_testing: bool
            Use fake dataset when True

    Returns:
        is_testing flag, DataFrame
    """
    if is_testing:
        return False, load_dataset(is_testing=True, sheet_name=sheet_name)
    return True, load_dataset(is_testing=False, sheet_name=sheet_name)


def extract_channel_data(df: pd.DataFrame, channel_num: int) -> pd.DataFrame:
    """
    Extract data for a single channel from long format.

    Long format: | batch | channel | length | ... | dim_error | flow_rate |

    Each row is one channel. Filter by channel number.

    Args:
        df: DataFrame with 'channel' column
        channel_num: Which channel to extract (1-4)

    Returns:
        DataFrame with only rows for that channel
    """
    if "channel" not in df.columns:
        raise ValueError("Expected long format with 'channel' column")

    return df[df["channel"] == channel_num].copy()
