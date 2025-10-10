import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from bayes_opt import BayesianOptimization

class ContextualBayesOpt:

    """
    constructor:
        dataset

    helper methods
        compute mean and std 
        compute accquisition func (ucb)

    methods:
        retrain surrogate 
        check drift 
        run bayes opt 

    """

    def __init__(self, pipeline, pbounds, lam=2.0):
        self.model = pipeline # sklearn pipeline
        self.pbounds = pbounds # param search space
        self.lam = lam # exploration hyperparam
        self.pre = None
        self.rf = None
        self.EXPECTED = None
        self.c_t = None

    def train_surrogate(self, X, y, verbose=False):
        param_grid = {
            "rf__n_estimators": [100, 200, 300, 400, 500],
            "rf__max_depth": [5, 10, 15, 20],
            "rf__max_features": ["sqrt", "log2", None],
        }
        
        # find best hyperparameters
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=5, # num cross-validation folds
            scoring='neg_mean_squared_error', # metric 
            n_jobs=-1,
            verbose=1 # display progress during fitting
        )

        grid_search.fit(X, y)

        if verbose:
            print("Best parameters found:", grid_search.best_params_)
            print("Best cross-validation score:", grid_search.best_score_)

        self.model = grid_search.best_estimator_
        self._build_schema()
        return self.model

    def evalute_surrogate(self, X_test, y_test):
        print("Surrogate model R² on test set:", self.model.score(X_test, y_test))

    # helper functions 
    def _build_schema(self):
        self.pre = self.model.named_steps["preprocess"]
        self.rf  = self.model.named_steps["rf"]
        self.EXPECTED = list(self.pre.feature_names_in_)

    def _make_row(self, c_t, x):
        # x contains knobs with strings, e.g. {"resin_type":"Resin_A", ...}
        row = {**c_t, **x}  # merge current context + knobs
        return pd.DataFrame([row], columns=self.EXPECTED) # EXPECTED from prerpocesser

    def _mu_sigma(self, df):
        """
        Inputs dataframe of candidates with knobs + context 
        Compute y_pred per tree in forest
        Outputs arrays for mean and std, per row
        """
        Xt = self.pre.transform(df)
        preds = np.vstack([t.predict(Xt) for t in self.rf.estimators_])  # [n_trees, n_samples]
        return preds.mean(axis=0), preds.std(axis=0, ddof=1)


    def _compute_ucb(self, x, lam=2.0):
        """
        x : dict
            Full input row (knobs + current context).
        lam : float
            Exploration weight λ.
        Returns
        -------
        float : UCB(x; λ)
        """
        import pandas as pd
        X1 = pd.DataFrame([x], columns=self.EXPECTED)   # enforce schema
        mu, sig = self._mu_sigma(X1)
        return mu[0] + lam * sig[0]

    def objective_ucb(self, **x):
        row = self._make_row(self.c_t, x)            # merges knobs and current context
        mu, sig = self._mu_sigma(row)
        ucb = mu[0] + self.lam * sig[0]
        return -ucb                   # BO maximizes (equiv to minimizing UCB)


    def compute_bayes_opt(self, c_t, verbose=False):
        self.c_t = c_t # set current context for objective
        optimizer = BayesianOptimization(
            f=self.objective_ucb,
            pbounds=self.pbounds,
            random_state=42,
        )

        optimizer.maximize(init_points=5, n_iter=20)

        best_result = optimizer.max
        best_params = best_result["params"]
        best_ucb = -best_result["target"]

        if verbose:
            print("Best parameters found:")
            print(best_params)

            print("\nBest UCB score:")
            print(best_ucb)
        
        return best_params, best_ucb, optimizer

