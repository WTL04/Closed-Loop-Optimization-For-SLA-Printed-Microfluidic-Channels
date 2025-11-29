import pandas as pd
from typing import Optional
from lab_runner import LabRunner

from ax.core import (
    SearchSpace,
    Experiment,
    Metric,
    Objective,
    OptimizationConfig,
    ParameterType,
)
from ax.core.arm import Arm
from ax.core.data import Data
from ax.adapter.registry import Generators
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import MinTrials
from ax.service.utils.instantiation import FixedFeatures


# TODO: Implement CBO with AX platform
class ContextualBayesOptAx:
    """
    Contextual Bayesian Optimization using Ax Dev (Github Version, most updated)

    - Uses Ax's internal surrogate (GP based) on knobs + context
    - Supports warm starting from an offline dataset
    - For each context snapshot c_t, suggests knob settings x that minimize a metric
    - After each experiment, appends (x, c_t, y) and Ax retrains internally
    """

    def __init__(
        self,
        search_space: SearchSpace,
        metric_name: str = "flow_rate_per_min",
        minimize: bool = True,
        generation_strategy: Optional[GenerationStrategy] = None,
        experiment_name: str = "cbo",
    ):
        """
        Args:
            search_space: ax.core.SearchSpace
                Includes both knob and context parameters.
            metric_name: str
                Name of the metric to optimize (column in your data).
            minimize: bool
                If True, minimize the metric. If False, maximize.
            generation_strategy: GenerationStrategy, optional
                Custom Ax generation strategy. If None, use Sobol + GPEI.
            experiment_name: str
                Name for the Ax Experiment.
        """

        self.search_space = search_space
        self.metric_name = metric_name
        self.minimize = minimize
        self.runner = LabRunner()

        # initialize Ax experiment with objective configuration
        self.experiment = Experiment(
            name=experiment_name,
            search_space=self.search_space,
            optimization_config=OptimizationConfig(
                objective=Objective(
                    metric=Metric(name=self.metric_name), minimize=self.minimize
                )
            ),
            runner=self.runner,
        )

        # GenerationStrategy: Sobol warmup -> BoTorch modular GP BO
        if generation_strategy is None:
            bo_specs = [
                GeneratorSpec(
                    generator_enum=Generators.BOTORCH_MODULAR,
                    model_kwargs={},  # default surrogate
                    model_gen_kwargs={},  # default candidate gen
                )
            ]
            bo_node = GenerationNode(
                name="BoTorch",
                generator_specs=bo_specs,
            )

            sobol_specs = [
                GeneratorSpec(
                    generator_enum=Generators.SOBOL,
                    model_kwargs={"seed": 42},
                )
            ]
            sobol_node = GenerationNode(
                name="Sobol",
                generator_specs=sobol_specs,
                transition_criteria=[
                    MinTrials(
                        threshold=5,  # after 5 completed trials
                        transition_to=bo_node.name,
                        use_all_trials_in_exp=True,
                    )
                ],
            )

            # wrap up generation strategy with nodes and steps
            self.generation_strategy = GenerationStrategy(
                name="Sobol+BoTorch", nodes=[sobol_node, bo_node]
            )

        else:
            # use custom generation_strategy strategy
            self.generation_strategy = generation_strategy

    def add_historical(self, df: pd.DataFrame):
        """
        Attach historical (x, c, y) data to the Ax experiment.

        df must contain:
            - one column per parameter in search_space.parameters.keys()
            - one column with name self.metric_name for the target
        """
        param_names = list(self.search_space.parameters.keys())
        records = []

        for row in df.itertuples(index=False):
            # ensure data types persist
            params = {}
            for name in param_names:
                p = self.search_space.parameters[name]
                val = getattr(row, name)

                if p.parameter_type is ParameterType.INT:
                    params[name] = int(val)
                elif p.parameter_type is ParameterType.FLOAT:
                    params[name] = float(val)
                else:
                    params[name] = val

            arm = Arm(parameters=params)

            # initilaize an trial and add metrics of trial into records
            trial = self.experiment.new_trial()
            trial.add_arm(arm)

            metric_val = getattr(row, self.metric_name)

            records.append(
                {
                    "trial_index": trial.index,
                    "arm_name": arm.name,
                    "metric_name": self.metric_name,
                    "metric_signature": self.metric_name,
                    "mean": float(metric_val),
                    "sem": 0.0,
                }
            )

        # attach metric values from each trial for surrogate to train on
        data = Data(df=pd.DataFrame.from_records(records))

        # append Data object into experiment Dataframe
        self.experiment.attach_data(data)

    def suggest(self, c_t: dict) -> dict:
        """Suggests knob settings given a context snapshot

        Args:
            c_t: dict
                Dictionary of context snapshot

        Returns:
            dict
                Suggested paramterization (knobs + fixed context)
        """

        # TODO: adjust suggest(c_t) with new version of generation_strategy
        # context = FixedFeatures(parameters=c_t)

        # generator_run = self.generation_strategy.gen(
        #     experiment=self.experiment,
        #     fixed_features=context,
        # )
        #
        # trial = self.experiment.new_trial(generator_run)
        # trial.mark_running()
        #
        # arm = trial.arms[0]  # pick first arm
        # return {"trial": trial, "params": arm.parameters}

    def observe(self, trial, metric_value: float):
        """
        Record the observed metric for a trial and mark it completed

        Args:
            trial : ax.core.trial.trial
                Trial returned by suggest
            metric_value : float
                Observed value of metric (e.g CV) for that trial
        """
        arm = trial.arms[0]

        df = pd.DataFrame(
            [
                {
                    "trial_index": trial.index,
                    "arm_name": arm.name,
                    "metric_name": self.metric_name,
                    "mean": float(metric_value),
                    "sem": 0.0,
                }
            ]
        )
        data = Data(df=df)
        self.experiment.attach_data(data)
        trial.mark_completed()

    def best_point(self):
        """
        Return the best parameterization according to the current Ax surrogate.

        Uses the model inside the GenerationStrategy, so you must have:
        - run suggest() at least once,
        - attached some observations via add_historical() and/or observe().

        Returns:
            dict with:
                - "params": best arm parameters (dict)
                - "mean": predicted mean metric at that point
                - "sem": predicted SEM at that point
        """
        # TODO: implement getting the best point using github version of Ax
