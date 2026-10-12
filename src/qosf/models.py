"""Forecasting models: two naive baselines, a linear model, an MLP and LightGBM."""
import lightgbm as lgb
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE


class ColumnBaseline:
    """Predicts t+h with one column observed at t (no training)."""

    def __init__(self, column: str):
        self.column = column

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[self.column].values


def lgbm(objective="regression", alpha=None, n_estimators=600):
    params = dict(n_estimators=n_estimators, learning_rate=0.03, num_leaves=63,
                  min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, random_state=RANDOM_STATE,
                  n_jobs=2, verbose=-1, objective=objective)
    if alpha is not None:
        params["alpha"] = alpha
    return lgb.LGBMRegressor(**params)


def log_target(model):
    """Train on log(1+y): throughput is heavy-tailed (0 to 530 Mbit/s)."""
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)


def make_models():
    return {
        "Persistence": ColumnBaseline("thr_lag0"),
        "Harmonic mean (5 s)": ColumnBaseline("thr_hm5"),
        "Ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)),
        "MLP": log_target(make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3,
                         max_iter=200, early_stopping=True, random_state=RANDOM_STATE))),
        "LightGBM": lgbm(),
    }
