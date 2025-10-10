import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from bayes_opt import BayesianOptimization


class CBO:

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

    def __init__(self, df, pipeline, bounds, lam=2.0):
        self.df = df.copy()
        self.model = pipeline
        self.bounds = bounds
        self.lam = lam

    # helper functions 
    def _build_schema(self):
        self.pre = self.model.named_steps["preprocess"]
        self.rf  = self.model.named_steps["rf"]
        self.EXPECTED = list(self.pre.feature_names_in_)



