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
    load_historical: bool = True,
    case_dir: str = "cfd/channelCase",
):
    """
    Run CBO using data from Google Sheets.

    Args:
        sheet_name: Name of the Google Sheet tab
        channel_num: Channel number to run. If None, auto-detects from latest in sheet.
        context: Context parameters. If None, prompts user interactively.
        append_to_sheets: If True, appends results to Google Sheets after running.
        cbo: Existing CBO instance. If None, creates new one.
        load_historical: If True, loads all historical data from sheets. Set to False
            when using a CBO that already has historical data loaded (e.g., from saved state).

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

    # load data from google sheets and get all history up to current channel
    use_real_data = True
    if load_historical:
        df_historical = pullData(sheet_name=sheet_name, verbose=False)
        cbo.add_historical(df_historical)
        print(f"Loaded {len(df_historical)} rows (channels 1-{channel_num})")
    else:
        print("Using existing CBO state (skipping historical data load)")

    # Run CBO for ONE channel
    result = run_single_channel(
        cbo,
        context,
        sheet_name,
        channel_num,
        use_real_data=use_real_data,
        case_dir=case_dir,
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

    # load fake data and get all history up to current channel
    is_testing, df_full = load_data_source(sheet_name=sheet_name, is_testing=True)
    use_real_data = False

    if "channel" in df_full.columns:
        df_historical = df_full[df_full["channel"] <= channel_num].copy()
    else:
        df_historical = df_full

    cbo.add_historical(df_historical)
    print(f"Loaded {len(df_historical)} rows (channels 1-{channel_num})")

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


def run_sequential(
    sheet_name: str = "Experiment Realistic Deltas",
    num_channels: int = 1,
    temp: str = "hot",
    layer_thickness_um: int = 100,
    testing: bool = False,
    append_to_sheets: bool = True,
    start_ambient: float = 80.0,
    start_resin_age: float = 1.0,
    resin_temp: float = 80.0,
    case_dir: str = "cfd/channelCase",
):
    """
    Run CBO sequentially for N channels (1→2→3→...).

    Args:
        sheet_name: Name of the Google Sheet tab
        num_channels: Number of channels to run sequentially
        temp: "cold" (ambient decreases) or "hot" (ambient increases)
        layer_thickness_um: 50 or 100 - layer thickness for all channels
        testing: If True, use interactive prompts; if False, use context_overtime
        append_to_sheets: If True, appends results to Google Sheets after each run
        start_ambient: Starting ambient temp in F for first channel (default 80)
        start_resin_age: Starting resin age in hours for first channel (default 1)
        resin_temp: Base resin temp in F (default 80)

    Returns:
        list of dicts with results for each channel
    """

    results = []

    # Choose cbo state file depending on which google sheet dataset
    if sheet_name == "Experiment Realistic Deltas":
        save_path = "contextual_opt/src/data/cbo_state_realistic.json"
    elif sheet_name == "Experiment Random Deltas":
        save_path = "contextual_opt/src/data/cbo_state_random.json"
    else:
        raise ValueError(f"Unknown sheet_name: {sheet_name}")

    cbo = ContextualBayesOptAx(
        search_space=build_search_space(),
        metric_name="dim_error",
        minimize=True,
        tracking_metrics=["flow_rate"],
    )

    if os.path.exists(save_path):
        print(f"Loading existing CBO state from {save_path}")
        cbo.load(save_path)
        print(f"Loaded {len(cbo.client._experiment.trials)} existing trials")
    else:
        # Load all available historical data once at the start
        df_full = pullData(sheet_name=sheet_name, verbose=False)
        if not df_full.empty:
            cbo.add_historical(df_full)
            print(f"Loaded initial {len(df_full)} historical rows")

    print(f"Starting sequential run - {num_channels} channels")
    print(f"Temp direction: {temp}, Layer thickness: {layer_thickness_um} µm")

    latest_channel = get_latest_col_value(column_name="channel", sheet_name=sheet_name)
    current_ambient = start_ambient
    current_resin_age = start_resin_age
    print(f"Start ambient: {start_ambient}°F, Start resin age: {start_resin_age}hr")
    next_channel = (int(latest_channel) + 1) if latest_channel else 1

    for i in range(num_channels):
        channel_num = next_channel + i

        print(f"\n{'=' * 60}")
        print(f"Channel {channel_num}")
        print(f"{'=' * 60}")

        if testing:
            context = get_context_snapshot()
        else:
            context = context_overtime(
                temp=temp,
                layer_thickness_um=layer_thickness_um,
                testing=False,
                start_ambient=current_ambient,
                start_resin_age=current_resin_age,
                resin_temp=resin_temp,
            )[0]

        current_ambient = context["ambient_temp"]
        current_resin_age = context["resin_age"] + 6

        print(
            f"Context: ambient={context['ambient_temp']}°F, resin={context['resin_temp']}°F, age={context['resin_age']}hr"
        )

        result = run_with_google_sheets(
            sheet_name=sheet_name,
            channel_num=channel_num,
            context=context,
            append_to_sheets=append_to_sheets,
            cbo=cbo,
            load_historical=False,
            case_dir=case_dir,
        )
        results.append(result)

        cbo.save(save_path)
        print(f"Saved CBO state after channel {channel_num}")

    print(f"\n{'=' * 60}")
    print(f"Completed {num_channels} channels!")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("CBO Sequential Runner - Interactive Setup")
    print("=" * 60)

    print("\n1) run_with_google_sheets (single channel)")
    print("2) run_with_testing (single channel, fake data)")
    print("3) run_sequential (multiple channels)")
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
        sheet_name = input(
            "Enter test sheet name (default: Experiment Random Deltas): "
        ).strip()
        if not sheet_name:
            sheet_name = "Experiment Random Deltas"

        result = run_with_testing(channel_num=channel_num, sheet_name=sheet_name)

    # sequential run, google sheets
    elif mode == "3":
        sheet_name = input(
            "Enter Google Sheet name (default: Experiment Realistic Deltas): "
        ).strip()
        if not sheet_name:
            sheet_name = "Experiment Realistic Deltas"

        num_channels_input = input("Number of channels (default: 1): ").strip()
        num_channels = int(num_channels_input) if num_channels_input else 1

        print("\nTemperature direction:")
        print("1) hot (ambient increases)")
        print("2) cold (ambient decreases)")
        temp_choice = input("Select (1-2, default: 1): ").strip()
        temp = "cold" if temp_choice == "2" else "hot"

        layer_choice = input("\nLayer thickness (50 or 100, default: 100): ").strip()
        layer_thickness_um = int(layer_choice) if layer_choice in ["50", "100"] else 100

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

        start_ambient_input = input("\nStart ambient temp (default: 80): ").strip()
        start_ambient = float(start_ambient_input) if start_ambient_input else 80.0

        start_resin_age_input = input("Start resin age in hours (default: 1): ").strip()
        start_resin_age = float(start_resin_age_input) if start_resin_age_input else 1.0

        resin_temp_input = input("Resin temp (default: 80): ").strip()
        resin_temp = float(resin_temp_input) if resin_temp_input else 70.0

        print("\n" + "=" * 60)
        print("Starting sequential run with:")
        print(f"  Sheet: {sheet_name}")
        print(f"  Channels: {num_channels}")
        print(f"  Temp: {temp}")
        print(f"  Layer thickness: {layer_thickness_um} µm")
        print(f"  Testing: {testing}")
        print(f"  Append to sheets: {append_to_sheets}")
        print(f"  Start ambient: {start_ambient}")
        print(f"  Start resin age: {start_resin_age}hr")
        print(f"  Resin temp: {resin_temp}")
        print("=" * 60)

        results = run_sequential(
            sheet_name=sheet_name,
            num_channels=num_channels,
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
