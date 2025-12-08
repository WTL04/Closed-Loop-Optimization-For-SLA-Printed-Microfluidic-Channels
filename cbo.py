import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from bayes_opt import BayesianOptimization


class ContextualBayesOpt:
    """
    Contextual Bayesian Optimization (CBO)
    --------------------------------------

    A closed-loop optimization framework that:
      - Trains a surrogate model f̂(x, c) → y (e.g., Random Forest Regressor)
      - Computes an acquisition function (UCB) that balances exploration/exploitation
      - Runs Bayesian Optimization over tunable knobs under a given context snapshot

    Attributes
    ----------
    model : sklearn.Pipeline
        Pre-built pipeline with preprocessing and regressor steps.
    pbounds : dict
        Parameter bounds for Bayesian Optimization (search space for knobs).
    lam : float
        Exploration-exploitation parameter λ for the UCB function.
    pre : object
        Fitted preprocessing transformer from the pipeline.
    rf : object
        Fitted RandomForestRegressor from the pipeline.
    EXPECTED : list
        Ordered list of feature names expected by the preprocessor.
    c_t : dict
        Current context snapshot before a print/experiment.
    """

    def __init__(self, pipeline, pbounds, lam=2.0):
        """
        Initialize the CBO class with a model pipeline, search space, and UCB parameter.

        Parameters
        ----------
        pipeline : sklearn.Pipeline
            The surrogate model pipeline containing preprocessing and regressor.
        pbounds : dict
            The tunable parameter search space for Bayesian Optimization.
        lam : float, optional (default=2.0)
            The exploration weight λ for the UCB acquisition function.
        """
        self.model = pipeline
        self.pbounds = pbounds
        self.lam = lam
        self.pre = None
        self.rf = None
        self.EXPECTED = None
        self.c_t = None

    # -------------------------------------------------------------------------
    # Surrogate model training and evaluation
    # -------------------------------------------------------------------------

    def train_surrogate(self, X, y, verbose=False):
        """
        Fit the surrogate model using grid search to tune RandomForest hyperparameters.

        Parameters
        ----------
        X : pd.DataFrame
            Training features (knobs + context variables).
        y : pd.Series or np.array
            Target variable (e.g., flow-rate CV).
        verbose : bool, optional
            Whether to print training diagnostics.

        Returns
        -------
        sklearn.Pipeline
            Trained surrogate pipeline with optimal hyperparameters.
        """
        param_grid = {
            "rf__n_estimators": [100, 200, 300, 400, 500],
            "rf__max_depth": [5, 10, 15, 20],
            "rf__max_features": ["sqrt", "log2", None],
        }

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        self._build_schema()

        if verbose:
            print("Best parameters found:")
            print(grid_search.best_params_)
            print("\nBest cross-validation score:")
            print(grid_search.best_score_)

        return self.model

    def evaluate_surrogate(self, X_test, y_test):
        """
        Evaluate surrogate performance on a held-out test set.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series or np.array
            True target values.

        Returns
        -------
        None
        """
        print("Surrogate model R² on test set:", self.model.score(X_test, y_test))

    # -------------------------------------------------------------------------
    # Internal helper functions
    # -------------------------------------------------------------------------

    def _build_schema(self):
        """Extract preprocessing and model handles from the fitted pipeline."""
        self.pre = self.model.named_steps["preprocess"]
        self.rf = self.model.named_steps["rf"]
        self.EXPECTED = list(self.pre.feature_names_in_)

    def _make_row(self, c_t, x):
        """
        Merge current context and candidate knobs into a single-row DataFrame.

        Parameters
        ----------
        c_t : dict
            Current context variables (ambient temp, resin temp, resin age, etc.).
        x : dict
            Candidate tunable parameters (knobs) proposed by the optimizer.

        Returns
        -------
        pd.DataFrame
            One-row DataFrame with the correct schema for prediction.
        """
        if self.EXPECTED is None:
            self._build_schema()
        row = {**c_t, **x}

        # Ensure categorical fields exist and are strings
        if "resin_type" not in row or pd.isna(row["resin_type"]):
            row["resin_type"] = "Resin_A"
        if "support_mode" not in row or pd.isna(row["support_mode"]):
            row["support_mode"] = "auto"

        return pd.DataFrame([row], columns=self.EXPECTED)

    def _mu_sigma(self, df):
        """
        Compute mean and standard deviation of predictions across all trees.

        Parameters
        ----------
        df : pd.DataFrame
            Input sample(s) with knobs + context features.

        Returns
        -------
        tuple (np.ndarray, np.ndarray)
            Mean and standard deviation of predictions.
        """
        if self.pre is None or self.rf is None:
            raise RuntimeError(
                "Surrogate not initialized. Call train_surrogate() or set a fitted pipeline."
            )

        Xt = self.pre.transform(df)
        preds = np.vstack([t.predict(Xt) for t in self.rf.estimators_])
        mu = preds.mean(axis=0)
        sig = preds.std(axis=0, ddof=1)
        sig[sig < 1e-9] = 1e-9  # numerical stability
        return mu, sig

    def _compute_ucb(self, x, c_t=None, lam=None):
        """
        Compute the UCB acquisition value for a given input sample.

        Parameters
        ----------
        x : dict
            Combined knob + context dictionary.
        lam : float
            Exploration weight λ.

        Returns
        -------
        float
            Upper Confidence Bound (UCB) value.
        """
        if c_t is None:
            if self.c_t is None:
                raise ValueError("Current context c_t is not set.")
            c_t = self.c_t

        if lam is None:
            lam = self.lam

        row = self._make_row(c_t, x)
        mu, sig = self._mu_sigma(row)
        return mu[0] + lam * sig[0]

    # -------------------------------------------------------------------------
    # Optimization logic
    # -------------------------------------------------------------------------

    def objective_ucb(self, **x):
        """
        Objective function for Bayesian Optimization.

        Combines model mean and uncertainty via UCB and returns the negative value
        because the optimizer maximizes the provided function.

        Parameters
        ----------
        **x : dict
            Candidate knobs from the optimizer.

        Returns
        -------
        float
            Negative UCB score.
        """
        ucb = self._compute_ucb(x)
        return -ucb  # minimize flow error

    def compute_bayes_opt(self, c_t, init_points=5, n_iter=20, verbose=False):
        """
        Run Bayesian Optimization loop to find optimal knob settings.

        Parameters
        ----------
        c_t : dict
            Current context snapshot before the experiment.
        verbose : bool, optional
            Whether to print optimization progress and best results.

        Returns
        -------
        tuple
            (best_params, best_ucb, optimizer)
        """
        self.c_t = c_t

        optimizer = BayesianOptimization(
            f=self.objective_ucb,  # function reference
            pbounds=self.pbounds,
            random_state=42,
        )

        optimizer.maximize(init_points=init_points, n_iter=n_iter)

        best_result = optimizer.max
        best_params = best_result["params"]
        best_neg_ucb = best_result["target"]  # maximized value of (-UCB)
        best_ucb = (
            -best_neg_ucb
        )  # turned positive for analysis, higher ucb here means better

        if verbose:
            print("Best parameters found:")
            print(best_params)
            print("\nBest UCB score:")
            print(best_ucb)

        return best_params, best_ucb, optimizer
