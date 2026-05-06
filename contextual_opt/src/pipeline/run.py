"""
Main orchestration for running CBO.

Entry point functions:
- run_with_google_sheets: Run single channel with real data
- run_with_testing: Run single channel with fake data
- run_batch_google_sheets: Run all 4 channels sequentially
"""

from contextual_opt.src.core.ax_cbo import ContextualBayesOptAx
from contextual_opt.src.api.sheets_api import get_latest_col_value, pullData
from contextual_opt.src.pipeline.data_loader import (
    load_data_source,
    extract_channel_data,
)

from contextual_opt.src.pipeline.search_space import build_search_space
from contextual_opt.src.pipeline.context import get_context_snapshot, context_overtime
from contextual_opt.src.pipeline.runner import run_single_channel
from contextual_opt.src.pipeline.utils import append_single_to_sheets
import os


def run_with_google_sheets(
    sheet_name: str,
    channel_num: int = None,
    context: dict = None,
    append_to_sheets: bool = True,
    cbo=None,
):
    """
    Run CBO using data from Google Sheets.

    Args:
        sheet_name: Name of the Google Sheet tab
        channel_num: Channel number to run. If None, auto-detects from latest in sheet.
        context: Context parameters. If None, prompts user interactively.
        append_to_sheets: If True, appends results to Google Sheets after running.

    Returns:
        dict with channel_results and any other relevant info
    """
    # get latest channel number from sheet if not provided
    if channel_num is None:
        channel_num = get_latest_col_value(column_name="channel", sheet_name=sheet_name)
        if channel_num is not None:
            channel_num = int(channel_num)
    print(f"Starting optimization on channel {channel_num}")

    # get context (interactive if not provided)
    if context is None:
        context = get_context_snapshot()

    # use existing cbo or initialize new one
    if cbo is None:
        cbo = ContextualBayesOptAx(
            search_space=build_search_space(),
            metric_name="dim_error",
            minimize=True,
            tracking_metrics=["flow_rate"],
        )

    # load data from google sheets and filter by channel
    df_full = pullData(sheet_name=sheet_name, verbose=False)
    use_real_data = True
    df_channel = extract_channel_data(df_full, channel_num)
    cbo.add_historical(df_channel)
    print(f"Loaded {len(df_channel)} rows for channel {channel_num}")

    # Run CBO for ONE channel
    result = run_single_channel(
        cbo, context, sheet_name, channel_num, use_real_data=use_real_data
    )

    # Append result to sheets if requested
    if append_to_sheets:
        append_single_to_sheets(result, context, sheet_name)

    print(f"\nCompleted channel {channel_num}")

    return {
        "channel_num": channel_num,
        "result": result,
        "context": context,
    }


def run_with_testing(
    channel_num: int = None,
    context: dict = None,
    sheet_name: str = "Experiment Realistic Deltas",
):
    """
    Run CBO using fake/testing data.

    Args:
        channel_num: Channel number to run. If None, auto-detects from latest in test data.
        context: Context parameters. If None, uses default values.
        sheet_name: Name of the fake dataset to use.

    Returns:
        dict with channel_results and any other relevant info
    """
    # Get latest channel from test data if not provided
    if channel_num is None:
        is_testing, df_full = load_data_source(sheet_name=sheet_name, is_testing=True)
        if "channel" in df_full.columns:
            channel_num = int(df_full["channel"].max())
        else:
            channel_num = 1

    # Use default context if not provided
    if context is None:
        context = {
            "layer_thickness_um": 100,
            "ambient_temp": 80.0,
            "resin_temp": 80.0,
            "resin_age": 15.0,
        }

    print(f"Running testing mode on channel {channel_num}")

    # initialize cbo
    cbo = ContextualBayesOptAx(
        search_space=build_search_space(),
        metric_name="dim_error",
        minimize=True,
        tracking_metrics=["flow_rate"],
    )

    # load fake data and filter by channel
    is_testing, df_full = load_data_source(sheet_name=sheet_name, is_testing=True)
    use_real_data = False

    if "channel" in df_full.columns:
        df_channel = extract_channel_data(df_full, channel_num)
    else:
        df_channel = df_full

    cbo.add_historical(df_channel)
    print(f"Loaded {len(df_channel)} rows for channel {channel_num}")

    # run cbo for one channel
    result = run_single_channel(
        cbo, context, sheet_name, channel_num, use_real_data=use_real_data
    )

    print(f"\nCompleted channel {channel_num} (testing mode - no sheet update)")

    return {
        "channel_num": channel_num,
        "result": result,
        "context": context,
    }


def run_batch_google_sheets(
    sheet_name: str = "Experiment Realistic Deltas",
    num_batches: int = 1,
    temp: str = "hot",
    layer_thickness_um: int = 100,
    testing: bool = False,
    append_to_sheets: bool = True,
    start_ambient: float = 80.0,
    start_resin_age: float = 1.0,
    resin_temp: float = 80.0,
    save_path: str = "contextual_opt/src/data/cbo_state.json",
):
    """
    Run CBO for multiple batches (each batch = 4 channels) sequentially using Google Sheets data.

    Args:
        sheet_name: Name of the Google Sheet tab
        num_batches: Number of batches to run (each batch = 4 channels)
        temp: "cold" (ambient decreases) or "hot" (ambient increases)
        layer_thickness_um: 50 or 100 - layer thickness for all channels/batches
        testing: If True, use interactive prompts; if False, use context_overtime
        append_to_sheets: If True, appends results to Google Sheets after each run
        start_ambient: Starting ambient temp in F for first batch (default 80)
        start_resin_age: Starting resin age in hours for first batch (default 1)
        resin_temp: Base resin temp in F (default 80)
        save_path: Path to save CBO state JSON file (auto-saved after each batch)

    Returns:
        list of dicts with results for each channel
    """
    results = []

    # Initialize CBO once for the entire batch run
    cbo = ContextualBayesOptAx(
        search_space=build_search_space(),
        metric_name="dim_error",
        minimize=True,
        tracking_metrics=["flow_rate"],
    )

    # Load existing state if available
    if os.path.exists(save_path):
        print(f"Loading existing CBO state from {save_path}")
        cbo.load(save_path)

    current_ambient = start_ambient
    current_resin_age = start_resin_age

    print(f"Starting {num_batches} batch(es) - {num_batches * 4} total channels")
    print(f"Temp direction: {temp}, Layer thickness: {layer_thickness_um} µm")
    print(f"Start ambient: {start_ambient}f, Start resin age: {start_resin_age}hr")
    print(f"CBO state will be saved to {save_path} after each batch")

    for batch_idx in range(num_batches):
        # get latest channel number to know where to start this batch
        latest_channel = get_latest_col_value(
            column_name="channel", sheet_name=sheet_name
        )
        if latest_channel is not None:
            latest_channel = int(latest_channel)
        else:
            latest_channel = 0

        # determine which channels to run in this batch (cycle 1-4)
        channels_to_run = [((latest_channel + i) % 4) + 1 for i in range(4)]

        print(f"\n{'=' * 60}")
        print(f"Batch {batch_idx + 1}/{num_batches} - Channels: {channels_to_run}")
        print(f"{'=' * 60}")

        # generate contexts for this batch (4 channels)
        if testing:
            # use interactive prompts for each channel
            contexts = [get_context_snapshot() for _ in range(4)]
        else:
            # use automated context_overtime with continuous drift
            contexts = context_overtime(
                temp=temp,
                layer_thickness_um=layer_thickness_um,
                testing=False,
                start_ambient=current_ambient,
                start_resin_age=current_resin_age,
                resin_temp=resin_temp,
            )

        # Update resin age for this batch (already includes +6hr per channel in context_overtime)
        # The last context has the final resin_age, use that as start for next batch
        if contexts:
            current_resin_age = (
                contexts[-1]["resin_age"] + 6
            )  # +6 for next batch's first channel

        # Run each channel with its corresponding context
        for i, channel_num in enumerate(channels_to_run):
            context = contexts[i]
            print(f"\n--- Channel {channel_num} ---")
            print(
                f"Context: ambient={context['ambient_temp']}F, resin={context['resin_temp']}F, age={context['resin_age']}hr"
            )

            result = run_with_google_sheets(
                sheet_name=sheet_name,
                channel_num=channel_num,
                context=context,
                append_to_sheets=append_to_sheets,
                cbo=cbo,
            )
            results.append(result)

        # Save CBO state after each batch
        cbo.save(save_path)
        print(f"Saved CBO state after batch {batch_idx + 1}")

        # Update ambient for next batch (continue drift from last channel's ambient)
        if contexts:
            current_ambient = contexts[-1]["ambient_temp"]

    print(f"\n{'=' * 60}")
    print(f"Completed {num_batches} batch(es)! Processed {len(results)} total channels")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("CBO Batch Runner - Interactive Setup")
    print("=" * 60)

    print("\n1) run_with_google_sheets (single channel)")
    print("2) run_with_testing (single channel, fake data)")
    print("3) run_batch_google_sheets (multiple batches)")
    mode = input("Select mode (1-3): ").strip()

    # single channel, google sheets
    if mode == "1":
        sheet_name = input(
            "Enter Google Sheet name (default: Experiment Realistic Deltas): "
        ).strip()
        if not sheet_name:
            sheet_name = "Experiment Realistic Deltas"
        channel_num_input = input("Channel number (press Enter for auto): ").strip()
        channel_num = int(channel_num_input) if channel_num_input else None

        result = run_with_google_sheets(sheet_name=sheet_name, channel_num=channel_num)

    # single channel fake data
    elif mode == "2":
        channel_num_input = input("Channel number (press Enter for auto): ").strip()
        channel_num = int(channel_num_input) if channel_num_input else None
        sheet_name = input(
            "Enter test sheet name (default: Experiment Random Deltas): "
        ).strip()
        if not sheet_name:
            sheet_name = "Experiment Random Deltas"

        result = run_with_testing(channel_num=channel_num, sheet_name=sheet_name)

    # multiple batch, google sheets
    elif mode == "3":
        sheet_name = input(
            "Enter Google Sheet name (default: Experiment Realistic Deltas): "
        ).strip()
        if not sheet_name:
            sheet_name = "Experiment Realistic Deltas"

        num_batches_input = input("Number of batches (default: 1): ").strip()
        num_batches = int(num_batches_input) if num_batches_input else 1

        print("\nTemperature direction:")
        print("1) hot (ambient increases)")
        print("2) cold (ambient decreases)")
        temp_choice = input("Select (1-2, default: 1): ").strip()
        temp = "cold" if temp_choice == "2" else "hot"

        layer_choice = input("\nLayer thickness (50 or 100, default: 100): ").strip()
        layer_thickness_um = int(layer_choice) if layer_choice in ["50", "100"] else 100

        # interavtive vs automated context drifts
        testing_input = (
            input(
                "\nTesting mode? (y/n, default: n) - y: interactive prompts, n: automated drift: "
            )
            .strip()
            .lower()
        )
        testing = testing_input == "y"

        append_input = (
            input("Append results to sheets? (y/n, default: y): ").strip().lower()
        )
        append_to_sheets = append_input != "n"

        start_ambient_input = input("\nStart ambient temp in F (default: 80): ").strip()
        start_ambient = float(start_ambient_input) if start_ambient_input else 80.0

        start_resin_age_input = input("Start resin age in hours (default: 1): ").strip()
        start_resin_age = float(start_resin_age_input) if start_resin_age_input else 1.0

        resin_temp_input = input("Resin temp in F (default: 80): ").strip()
        resin_temp = float(resin_temp_input) if resin_temp_input else 80.0

        print("\n" + "=" * 60)
        print("Starting batch run with:")
        print(f"  Sheet: {sheet_name}")
        print(f"  Batches: {num_batches}")
        print(f"  Temp: {temp}")
        print(f"  Layer thickness: {layer_thickness_um} µm")
        print(f"  Testing: {testing}")
        print(f"  Append to sheets: {append_to_sheets}")
        print(f"  Start ambient: {start_ambient}F")
        print(f"  Start resin age: {start_resin_age}hr")
        print(f"  Resin temp: {resin_temp}F")
        print("=" * 60)

        results = run_batch_google_sheets(
            sheet_name=sheet_name,
            num_batches=num_batches,
            temp=temp,
            layer_thickness_um=layer_thickness_um,
            testing=testing,
            append_to_sheets=append_to_sheets,
            start_ambient=start_ambient,
            start_resin_age=start_resin_age,
            resin_temp=resin_temp,
        )

    else:
        print("Invalid mode selected.")
