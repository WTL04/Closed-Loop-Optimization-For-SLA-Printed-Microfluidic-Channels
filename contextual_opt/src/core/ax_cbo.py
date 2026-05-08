import pandas as pd
from typing import Optional

from ax.api import Client
from ax.core import (
    Arm,
    GeneratorRun,
    SearchSpace,
    Experiment,
    ParameterType,
    OptimizationConfig,
    Objective,
    Metric,
    RangeParameter,
)
from ax.core.observation import ObservationFeatures

from contextual_opt.src.core.lab_runner import LabRunner


class ContextualBayesOptAx:
    """
    Contextual Bayesian Optimization using new Ax API (ax.api.Client).

    - Uses Client for experiment state preservation (trial history, not model state)
    - Supports contextual optimization via fixed_parameters
    - Save/load preserves experiment/trial data but NOT surrogate model parameters
    - Sequential optimization: trials run one at a time
    """

    def __init__(
        self,
        search_space: SearchSpace,
        metric_name: str = "dimensional_error",
        minimize: bool = True,
        experiment_name: str = "cbo",
        tracking_metrics: Optional[list[str]] = None,
    ):
        """
        Args:
            search_space: ax.core.SearchSpace
                Includes both knob and context parameters.
            metric_name: str
                Name of the metric to optimize.
            minimize: bool
                If True, minimize the metric. If False, maximize.
            experiment_name: str
                Name for the Ax Experiment.
            tracking_metrics: list[str], optional
                List of metric names to track.
        """
        self.search_space = search_space
        self.metric_name = metric_name
        self.minimize = minimize
        self.tracking_metrics = tracking_metrics or []
        self.runner = LabRunner()
        self._context_params = set()

        # create experiment with optimization config
        self.experiment = Experiment(
            name=experiment_name,
            search_space=self.search_space,
            optimization_config=OptimizationConfig(
                objective=Objective(
                    metric=Metric(name=metric_name),
                    minimize=minimize,
                )
            ),
        )

        # create Client and set experiment
        self.client = Client()
        self.client.set_experiment(self.experiment)

        # configure tracking metrics through client
        if self.tracking_metrics:
            self.client.configure_tracking_metrics(metric_names=self.tracking_metrics)

    def set_context_params(self, context_param_names: list[str]):
        """
        Set which parameters are context (fixed) vs. knobs (to be suggested).

        Args:
            context_param_names: List of parameter names that are context
        """
        self._context_params = set(context_param_names)

    def add_historical(self, df: pd.DataFrame):
        """
        Attach historical (x, c, y) data to the experiment.

        Args:
            df: DataFrame with columns for all parameters + metrics
        """
        param_names = list(self.search_space.parameters.keys())

        for row in df.itertuples(index=False):
            # Build parameters dict for all columns
            params = {}
            for name in param_names:
                try:
                    val = getattr(row, name)
                except AttributeError:
                    continue

                if (
                    val == ""
                    or val is None
                    or (hasattr(val, "__float__") and pd.isna(val))
                ):
                    val = 0.0

                p = self.search_space.parameters[name]
                if p.parameter_type == ParameterType.INT:
                    params[name] = int(float(val))
                elif p.parameter_type == ParameterType.FLOAT:
                    params[name] = float(val)
                else:
                    params[name] = val

            # Get metric values
            main_metric_val = getattr(row, self.metric_name, None)
            if (
                main_metric_val is None
                or main_metric_val == ""
                or (hasattr(main_metric_val, "__float__") and pd.isna(main_metric_val))
            ):
                main_metric_val = 0.0
            all_metrics = {self.metric_name: float(main_metric_val)}
            for tm in self.tracking_metrics:
                try:
                    tm_val = getattr(row, tm)

                    # ensure no empty cells
                    if tm_val is not None and not pd.isna(tm_val) and tm_val != "":
                        all_metrics[tm] = float(tm_val)
                except AttributeError:
                    pass

            # Convert to raw_data format, handling empty strings
            raw_data = {
                k: (float(v) if v != "" else 0.0) for k, v in all_metrics.items()
            }

            # Create trial directly with historical parameters, bypassing GenerationStrategy
            arm = Arm(parameters=params)
            gr = GeneratorRun(arms=[arm])
            trial = self.client._experiment.new_trial(generator_run=gr)
            trial.mark_running(no_runner_required=True)
            self.client.complete_trial(trial_index=trial.index, raw_data=raw_data)

        # Configure GS to skip Center+Sobol init, using historical trials as initialization
        if len(df) > 0:
            self.client.configure_generation_strategy(
                method="fast",
                initialization_budget=len(df),
                initialize_with_center=False,
                use_existing_trials_for_initialization=True,
            )

    def suggest(self, c_t: dict):
        """
        Suggests knob settings given a context snapshot.

        Args:
            c_t: dict - context dictionary with context parameter values

        Returns:
            dict with trial info and parameters
        """
        # Use context as fixed_parameters
        trial_dict = self.client.get_next_trials(
            max_trials=1,
            fixed_parameters=c_t,
        )

        trial_idx = list(trial_dict.keys())[0]
        parameters = trial_dict[trial_idx]

        trial = type(
            "Trial",
            (),
            {
                "trial_index": trial_idx,
                "arm": type("Arm", (), {"parameters": parameters}),
            },
        )()

        return {"trial": trial, "params": parameters}

    def observe(self, trial, metric_value: float = None, metric_values: dict = None):
        """
        Record the observed metric(s) for a trial and mark it completed.

        Args:
            trial: Trial returned by suggest
            metric_value: float - Observed value of primary metric
            metric_values: dict - Dict of metric_name -> value for multiple metrics
        """
        trial_index = trial.trial_index

        if metric_values:
            raw_data = {k: float(v) for k, v in metric_values.items()}
        elif metric_value is not None:
            raw_data = {self.metric_name: float(metric_value)}
        else:
            return

        self.client.complete_trial(trial_index=trial_index, raw_data=raw_data)
        print(f"Trial Status: Completed")

    def optimization_trace(self) -> pd.DataFrame:
        """
        Returns a DataFrame with trial optimization history.
        """
        try:
            df = self.client.get_trials_dataframe()
            if df.empty:
                return pd.DataFrame(columns=["trial_index", "mean", "best_so_far"])
            return df
        except Exception:
            return pd.DataFrame(columns=["trial_index", "mean", "best_so_far"])

    def save(self, filepath: str):
        """
        Save the Client state to JSON file.

        This preserves:
        - Experiment configuration
        - All trial data (historical + new)

        Note: Does NOT preserve surrogate model state (GP hyperparameters,
        kernel parameters, or learned weights). Only experiment/trial history
        is saved, allowing optimization to resume from where it stopped.

        Args:
            filepath: Path to save the JSON file.
        """
        import os

        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )
        self.client.save_to_json_file(filepath)
        print(f"Saved Client state to {filepath}")

    def load(self, filepath: str):
        """
        Load Client state from JSON file.

        Loads experiment/trial data. Note that the surrogate model is
        retrained on load, so model state is not preserved.

        Args:
            filepath: Path to the JSON file.

        Returns:
            self (for chaining)
        """
        import os

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"State file not found: {filepath}")

        self.client = Client.load_from_json_file(filepath)
        self.experiment = self.client._experiment
        print(f"Loaded Client state from {filepath}")

        return self
