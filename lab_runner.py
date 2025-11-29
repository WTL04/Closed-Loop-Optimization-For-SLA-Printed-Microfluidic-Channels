from typing import Any
from ax.core.runner import Runner
from ax.core.trial import Trial


class LabRunner(Runner):
    """
    Minimal runner for manual lab experiments.

    It does not actually launch anything. It just records that
    a trial with certain parameters was 'dispatched'.
    """

    def run(self, trial: Trial) -> dict[str, Any]:
        # You can inspect/log the parameters here if you want
        # For single-arm trials:
        arm = trial.arms[0]
        params = arm.parameters

        # This is where you could:
        # - send params to a lab notebook
        # - log to a file
        # - trigger equipment if you ever automate

        print(f"[LabRunner] Trial {trial.index} dispatched with params: {params}")

        # Return arbitrary metadata; Ax will store this on the trial
        return {
            "status": "dispatched",
            "note": "manual run in lab",
        }

