"""Input drift monitoring for the risk classifier.

Every spec processed by the generator is compared against statistics captured at
training time. The Population Stability Index (PSI) is computed for categorical
and binned features:

    PSI = sum_i (p_i - q_i) * ln(p_i / q_i)

where q is the training distribution and p the distribution in the new spec.
Common reading: < 0.1 stable, 0.1 to 0.25 moderate shift, > 0.25 significant shift.

Per spec, an alert is raised when the share of unseen vocabulary (OOV rate) is
high or the classifier's mean confidence drops: both mean the model is being asked
about APIs unlike its training data. PSI values are exported as metrics; the
CloudWatch alarms in aws/template.yaml evaluate them as daily aggregates.
This plays the role SageMaker Model Monitor's data-quality baseline plays for a
hosted endpoint, but it also works for the offline CLI.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LENGTH_BINS = [0, 5, 10, 20, 40, 80, 10_000]
METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OTHER"]
DEFAULT_BASELINE = Path(__file__).parent / "models" / "drift_baseline.json"
EPS = 1e-4
OOV_ALERT_FLOOR = 0.20


def _method_dist(X: pd.DataFrame) -> list[float]:
    m = X["method"].where(X["method"].isin(METHODS[:-1]), "OTHER")
    counts = m.value_counts()
    return [float(counts.get(k, 0)) / max(len(X), 1) for k in METHODS]


def _length_dist(X: pd.DataFrame) -> list[float]:
    lengths = X["text"].str.split().str.len().fillna(0)
    hist, _ = np.histogram(lengths, bins=LENGTH_BINS)
    return (hist / max(len(X), 1)).tolist()


def _vocabulary(model) -> set[str]:
    try:
        vec = model.named_steps["features"].named_transformers_["text"]
        return {t for t in vec.vocabulary_ if " " not in t}
    except (KeyError, AttributeError):
        return set()


def _oov_rate(X: pd.DataFrame, vocab: set[str]) -> float:
    if not vocab:
        return 0.0
    toks = [t for text in X["text"] for t in text.split()]
    return sum(t not in vocab for t in toks) / max(len(toks), 1)


def psi(expected: list[float], actual: list[float]) -> float:
    total = 0.0
    for q, p in zip(expected, actual):
        q, p = max(q, EPS), max(p, EPS)
        total += (p - q) * math.log(p / q)
    return round(total, 4)


def build_baseline(X_train: pd.DataFrame, model) -> dict[str, Any]:
    proba = model.predict_proba(X_train)
    return {
        "n": int(len(X_train)),
        "method_dist": _method_dist(X_train),
        "length_dist": _length_dist(X_train),
        "oov_rate": round(_oov_rate(X_train, _vocabulary(model)), 4),
        "mean_confidence": round(float(proba.max(axis=1).mean()), 4),
        "class_dist": dict(zip(model.classes_, np.bincount(proba.argmax(axis=1), minlength=3) / len(X_train))),
    }


def check_drift(X: pd.DataFrame, model, baseline_path: str | Path = DEFAULT_BASELINE) -> dict[str, Any]:
    path = Path(baseline_path)
    if model is None or not path.exists() or len(X) == 0:
        return {"status": "skipped"}
    base = json.loads(path.read_text())
    proba = model.predict_proba(X)
    report = {
        "psi_method": psi(base["method_dist"], _method_dist(X)),
        "psi_text_length": psi(base["length_dist"], _length_dist(X)),
        "oov_rate": round(_oov_rate(X, _vocabulary(model)), 4),
        "baseline_oov_rate": base["oov_rate"],
        "mean_confidence": round(float(proba.max(axis=1).mean()), 4),
        "baseline_mean_confidence": base["mean_confidence"],
    }
    alerts = []
    report["sample_size"] = int(len(X))
    # PSI of a single API against the whole training corpus is expected to be high (one API is
    # not a random sample of all APIs), so PSI is exported as a metric and alarmed on as a daily
    # aggregate in CloudWatch. Per-spec alerts use the two signals that do not depend on API style.
    if report["oov_rate"] > max(OOV_ALERT_FLOOR, base["oov_rate"] + 0.15):
        alerts.append("vocabulary drift: many unseen terms")
    if report["mean_confidence"] < base["mean_confidence"] - 0.10:
        alerts.append("confidence drop: review risk labels before deploying")
    report["status"] = "alert" if alerts else "ok"
    report["alerts"] = alerts
    return report
