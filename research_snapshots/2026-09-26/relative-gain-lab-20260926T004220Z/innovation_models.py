"""Fixed gain constructions for the relative-gain development/replay study.

Training losses are squared geometric errors in mm^2.  Inference accepts only
the observable feature array; no labels, case identity, or paths are accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)


NORMALIZATION_EPSILON_MM2 = 0.01
HGB_PARAMS = {
    "learning_rate": 0.08,
    "max_iter": 80,
    "max_leaf_nodes": 7,
    "max_depth": 3,
    "min_samples_leaf": 80,
    "l2_regularization": 1.0,
    "early_stopping": False,
}


def _features(X: np.ndarray, n_features: int | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("X must be a two-dimensional array with features")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"Expected {n_features} observable features, got {X.shape[1]}")
    # HGB handles missing features natively. Infinite features are invalid.
    if np.isinf(X).any():
        raise ValueError("X must not contain infinite values")
    return X


def _vector(value: np.ndarray, name: str, n: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (n,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of length {n}")
    return result


def _scores(value: np.ndarray, n: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (n,) or not np.isfinite(result).all():
        raise FloatingPointError("The fitted policy produced invalid scores")
    return result


@dataclass
class ConstantRegressor:
    value: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value, dtype=np.float64)


@dataclass
class ConstantProbability:
    probability: float

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        result = np.empty((len(X), 2), dtype=np.float64)
        result[:, 0] = 1.0 - self.probability
        result[:, 1] = self.probability
        return result


def _fit_regressor(
    X: np.ndarray, target: np.ndarray, weight: np.ndarray, seed: int
) -> Any:
    active = weight > 0
    X, target, weight = X[active], target[active], weight[active]
    if len(target) == 0:
        # An empty conditional branch contributes zero magnitude.
        return ConstantRegressor(0.0)
    if len(target) < 2 * HGB_PARAMS["min_samples_leaf"] or np.ptp(target) == 0:
        return ConstantRegressor(float(np.average(target, weights=weight)))
    model = HistGradientBoostingRegressor(
        loss="squared_error", random_state=seed, **HGB_PARAMS
    )
    return model.fit(X, target, sample_weight=weight)


def _fit_probability(
    X: np.ndarray, positive: np.ndarray, weight: np.ndarray, seed: int
) -> Any:
    active = weight > 0
    X, positive, weight = X[active], positive[active], weight[active]
    p = float(np.average(positive, weights=weight))
    if len(positive) < 2 * HGB_PARAMS["min_samples_leaf"] or np.unique(positive).size < 2:
        return ConstantProbability(p)
    model = HistGradientBoostingClassifier(
        loss="log_loss", random_state=seed, **HGB_PARAMS
    )
    return model.fit(X, positive.astype(np.int64), sample_weight=weight)


@dataclass
class RegressionPolicy:
    model: Any
    n_features: int
    default_threshold: float = 0.0

    def score(self, X: np.ndarray) -> np.ndarray:
        X = _features(X, self.n_features)
        if len(X) == 0:
            return np.empty(0, dtype=np.float64)
        return _scores(self.model.predict(X), len(X))


@dataclass
class BenefitHarmPolicy:
    benefit_model: Any
    harm_model: Any
    n_features: int
    default_threshold: float = 0.0

    def score(self, X: np.ndarray) -> np.ndarray:
        X = _features(X, self.n_features)
        if len(X) == 0:
            return np.empty(0, dtype=np.float64)
        benefit = np.maximum(self.benefit_model.predict(X), 0.0)
        harm = np.maximum(self.harm_model.predict(X), 0.0)
        return _scores(benefit - harm, len(X))


@dataclass
class HurdleGainPolicy:
    probability_model: Any
    positive_model: Any
    nonpositive_model: Any
    n_features: int
    default_threshold: float = 0.0

    def score(self, X: np.ndarray) -> np.ndarray:
        X = _features(X, self.n_features)
        if len(X) == 0:
            return np.empty(0, dtype=np.float64)
        p = np.clip(self.probability_model.predict_proba(X)[:, 1], 0.0, 1.0)
        magnitude_plus = np.maximum(self.positive_model.predict(X), 0.0)
        magnitude_minus = np.maximum(self.nonpositive_model.predict(X), 0.0)
        return _scores(p * magnitude_plus - (1.0 - p) * magnitude_minus, len(X))


def train_policies(
    X: np.ndarray,
    gain: np.ndarray,
    e0: np.ndarray,
    e1: np.ndarray,
    sample_weight: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit all four predeclared arms on development rows only.

    `gain = e0 - e1`, with e0/e1 in mm^2.  Every nonconstant HGB head
    uses HGB_PARAMS and the supplied seed. Weights are passed without
    rescaling; zero-weight rows cannot affect a fitted head. Zero gains
    belong to the nonpositive hurdle branch. All policies use strict
    score > threshold downstream; zero is the natural threshold.
    """
    X = _features(X)
    n, n_features = X.shape
    if n == 0:
        raise ValueError("At least one development training row is required")
    gain = _vector(gain, "gain", n)
    e0 = _vector(e0, "e0", n)
    e1 = _vector(e1, "e1", n)
    if (e0 < 0).any() or (e1 < 0).any():
        raise ValueError("e0 and e1 must be nonnegative squared errors in mm^2")
    weight = np.ones(n, dtype=np.float64) if sample_weight is None else _vector(
        sample_weight, "sample_weight", n
    )
    if (weight < 0).any() or not (weight > 0).any():
        raise ValueError("sample_weight must be nonnegative with positive total mass")
    denominator = e0 + e1 + NORMALIZATION_EPSILON_MM2
    normalized = gain / denominator
    if not np.isfinite(normalized).all() or not np.isfinite(denominator).all():
        raise ValueError("Squared errors must permit finite normalized-gain targets")
    positive = gain > 0.0
    direct = _fit_regressor(X, gain, weight, seed)
    relative = _fit_regressor(X, normalized, weight, seed)
    benefit = _fit_regressor(X, np.maximum(gain, 0.0), weight, seed)
    harm = _fit_regressor(X, np.maximum(-gain, 0.0), weight, seed)
    probability = _fit_probability(X, positive, weight, seed)
    magnitude_plus = _fit_regressor(X[positive], gain[positive], weight[positive], seed)
    magnitude_minus = _fit_regressor(
        X[~positive], -gain[~positive], weight[~positive], seed
    )
    return {
        "direct_gain": RegressionPolicy(direct, n_features),
        "normalized_gain": RegressionPolicy(relative, n_features),
        "benefit_harm": BenefitHarmPolicy(benefit, harm, n_features),
        "hurdle_gain": HurdleGainPolicy(
            probability, magnitude_plus, magnitude_minus, n_features
        ),
    }
