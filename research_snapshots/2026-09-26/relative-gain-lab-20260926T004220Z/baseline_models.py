"""Frozen-candidate selector baselines; inference consumes only cached X.

The absolute-confidence arm is a literature-inspired mechanism baseline, not
an official implementation or reproduction of a published stereo algorithm.
See BASELINE_NOTES.md for targets, feature semantics, and fairness limits.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


N_FEATURES = 64
FEATURE_INDICES = {
    "offset_0": 8,
    "training_cost_0": 12,
    "mode_gap_0": 14,
    "source_agreement_0": 20,
}
HGB_PARAMS = dict(
    learning_rate=0.08,
    max_iter=80,
    max_leaf_nodes=7,
    max_depth=3,
    min_samples_leaf=80,
    l2_regularization=1.0,
    early_stopping=False,
)
METHOD_NAMES = (
    "photo_cost",
    "mode_gap",
    "source_agreement",
    "small_displacement",
    "absolute_confidence_hgb",
    "candidate_error_hgb",
    "gain_sign_hgb",
)


def _features(X):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != N_FEATURES:
        raise ValueError(f"X must have shape (N, {N_FEATURES})")
    if not np.isfinite(X).all():
        raise ValueError("X must be finite cached A_all_post features")
    return X


def _vector(value, n, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (n,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite vector of length {n}")
    return value


def _finite_scores(value, n):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (n,) or not np.isfinite(value).all():
        raise ValueError("policy produced non-finite or incorrectly shaped scores")
    return value


@dataclass
class ScalarPolicy:
    """One observable feature; no training-dependent state or validity gate."""

    feature_index: int
    sign: float = 1.0
    absolute: bool = False
    default_threshold: float | None = None

    def score(self, X):
        X = _features(X)
        value = np.asarray(X[:, self.feature_index], dtype=np.float64)
        if self.absolute:
            value = np.abs(value)
        return _finite_scores(self.sign * value, len(X))


@dataclass
class ConstantPolicy:
    """Defined classifier output for a single positive-weight target class."""

    value: float
    default_threshold: float = 0.5

    def score(self, X):
        X = _features(X)
        return _finite_scores(np.full(len(X), self.value), len(X))


@dataclass
class ClassifierPolicy:
    estimator: Any
    default_threshold: float = 0.5

    def score(self, X):
        X = _features(X)
        if len(X) == 0:
            return np.empty(0, dtype=np.float64)
        positive = np.flatnonzero(self.estimator.classes_ == 1)
        if len(positive) != 1:
            raise ValueError("classifier must have exactly one positive target class")
        return _finite_scores(self.estimator.predict_proba(X)[:, positive[0]], len(X))


@dataclass
class CandidateErrorPolicy:
    estimator: Any
    default_threshold: float = -1.0

    def score(self, X):
        X = _features(X)
        if len(X) == 0:
            return np.empty(0, dtype=np.float64)
        # Do not clip predictions: the arm is exactly negative predicted e1.
        return _finite_scores(-self.estimator.predict(X), len(X))


def _fit_classifier(X, target, sample_weight, seed):
    active = np.ones(len(X), dtype=bool) if sample_weight is None else sample_weight > 0
    classes = np.unique(target[active])
    if len(classes) == 1:
        return ConstantPolicy(float(classes[0]))
    estimator = HistGradientBoostingClassifier(
        loss="log_loss", random_state=seed, **HGB_PARAMS
    )
    estimator.fit(X, target, sample_weight=sample_weight)
    return ClassifierPolicy(estimator)


def train_policies(X, gain, e0, e1, sample_weight, seed):
    """Fit seven baseline policies without threshold tuning.

    X is the finite, ordered 64-column A_all_post array. e0 and e1 are
    nonnegative squared geometric errors in mm^2, and gain is e0 - e1.
    Only supplied development rows may enter this function. Weights are passed
    unchanged to HGB; any case/scene balancing and normalization are the
    orchestrator's responsibility. Selection uses strict score > threshold.
    """
    X = _features(X)
    if len(X) == 0:
        raise ValueError("training requires at least one row")
    gain = _vector(gain, len(X), "gain")
    e0 = _vector(e0, len(X), "e0")
    e1 = _vector(e1, len(X), "e1")
    if np.any(e0 < 0) or np.any(e1 < 0):
        raise ValueError("e0 and e1 must be nonnegative squared errors")
    if not np.allclose(gain, e0 - e1, rtol=1e-6, atol=1e-6):
        raise ValueError("gain must equal e0 - e1 in mm^2")
    if sample_weight is not None:
        sample_weight = _vector(sample_weight, len(X), "sample_weight")
        if np.any(sample_weight < 0) or not np.any(sample_weight > 0):
            raise ValueError("sample_weight must be nonnegative with positive total mass")

    result = {
        "photo_cost": ScalarPolicy(FEATURE_INDICES["training_cost_0"], sign=-1.0),
        "mode_gap": ScalarPolicy(FEATURE_INDICES["mode_gap_0"]),
        "source_agreement": ScalarPolicy(FEATURE_INDICES["source_agreement_0"]),
        "small_displacement": ScalarPolicy(
            FEATURE_INDICES["offset_0"], sign=-1.0, absolute=True
        ),
        "absolute_confidence_hgb": _fit_classifier(
            X, (e1 <= 1.0).astype(np.int8), sample_weight, seed
        ),
    }
    error_model = HistGradientBoostingRegressor(
        loss="squared_error", random_state=seed, **HGB_PARAMS
    )
    error_model.fit(X, e1, sample_weight=sample_weight)
    result["candidate_error_hgb"] = CandidateErrorPolicy(error_model)
    result["gain_sign_hgb"] = _fit_classifier(
        X, (gain > 0.0).astype(np.int8), sample_weight, seed
    )
    assert tuple(result) == METHOD_NAMES
    return result
