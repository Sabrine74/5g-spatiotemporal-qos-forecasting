"""Conformalized quantile regression (Romano et al., NeurIPS 2019).

A pair of quantile models gives a raw interval [q_lo, q_hi]. A held-out calibration set
measures how much that interval must be widened so that the true value falls inside it
with probability >= 1 - alpha.
"""
import numpy as np

from .models import lgbm


class CQR:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def fit(self, X_fit, y_fit, X_cal, y_cal):
        self.lo_ = lgbm("quantile", alpha=self.alpha / 2, n_estimators=400).fit(X_fit, y_fit)
        self.hi_ = lgbm("quantile", alpha=1 - self.alpha / 2, n_estimators=400).fit(X_fit, y_fit)
        lo, hi = self.raw_interval(X_cal)
        scores = np.maximum(lo - y_cal, y_cal - hi)
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_ = float(np.quantile(scores, level, method="higher"))
        return self

    def raw_interval(self, X):
        lo, hi = self.lo_.predict(X), self.hi_.predict(X)
        return np.minimum(lo, hi), np.maximum(lo, hi)

    def predict_interval(self, X):
        lo, hi = self.raw_interval(X)
        return np.clip(lo - self.q_, 0, None), hi + self.q_
