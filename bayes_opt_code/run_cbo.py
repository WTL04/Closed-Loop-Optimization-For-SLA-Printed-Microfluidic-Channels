"""
run_cbo.py
----------
Single-shot Contextual Bayesian Optimization to generate suggested_parameters.json

CHECKLIST - Steps to complete:
===============================
[ ] 1. Get context snapshot (ambient_temp, resin_temp, resin_age)
[ ] 2. Define parameter space (pbounds) for Bayesian Optimization
[ ] 3. Load historical data from CSV/dataset
[ ] 4. Prepare training data (X = features, y = cv target)
[ ] 5. Build sklearn pipeline (preprocessing + RandomForestRegressor)
[ ] 6. Initialize CBO with pipeline and pbounds
[ ] 7. Train surrogate model (train_surrogate)
[ ] 8. Run Bayesian Optimization to suggest parameters
[ ] 9. Decode parameters from encoded values to physical values
[ ] 10. Save suggested parameters to suggested_parameters.json

Output: suggested_parameters.json
"""

import pandas as pd
from pathlib import Path 

def load_dataset(is_testing: bool, verbose=True):
    """
    Returns DataFrame from Google Spreadsheet or a chosen fake dataset.

    Args:
        is_testing: bool
            Uses fake dataset when True, Google Spreadsheet when False
    """
    if is_testing:
        choice = input(
            "Choose fake dataset: 1) dataset_5_batches.csv 2) dataset_10_batches.csv 3) dataset_15_batches.csv 4) dataset_30_batches.csv: "
        )

        if choice == "1":
            path = "../../datasets/dataset_5_batches.csv"
        elif choice == "2":
            path = "../../datasets/dataset_10_batches.csv"
        elif choice == "3":
            path = "../../datasets/dataset_15_batches.csv"
        elif choice == "4":
            path = "../../datasets/dataset_30_batches.csv"
        else:
            raise ValueError("Invalid fake dataset option")

        if verbose:
            print(f"Loading fake dataset: {path}")
        return pd.read_csv(path)

    # pull from google sheets api
    return pullData(sheet_name="Bayes Opt", verbose=verbose)





if __name__ == "__main__":

