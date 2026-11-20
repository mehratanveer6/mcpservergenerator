"""Operation risk classification (read / write / destructive).

The classifier decides three things in a generated server:
  * the MCP tool annotations (readOnlyHint, destructiveHint, idempotentHint)
  * whether a tool is enabled under the active safety policy
  * which tools are surfaced first in curated mode

A trained scikit-learn pipeline is loaded from mcpforge/models/risk_model.joblib.
If the artifact is missing, a transparent HTTP-method rule is used instead so the
generator still works offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .features import frame_from_operations
from .ir import Operation
from .rules import SPECIFIC_RULES, label_operation

LABELS = ("read", "write", "destructive")
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "risk_model.joblib"

# Safety bias: an operation is treated as destructive if P(destructive) reaches this
# value even when another class has a higher probability. Missing a destructive
# operation (false negative) is far more costly than an unnecessary confirmation.
DESTRUCTIVE_THRESHOLD = 0.35
LOW_CONFIDENCE = 0.60


def build_pipeline(C: float = 4.0, max_features: int = 20000, ngram_max: int = 2, min_df: int = 2):
    """Model definition used by ml/train.py (and by SageMaker script mode)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    features = ColumnTransformer(
        [
            ("method", OneHotEncoder(handle_unknown="ignore"), ["method"]),
            ("text", TfidfVectorizer(ngram_range=(1, ngram_max), max_features=max_features,
                                     min_df=min_df, sublinear_tf=True), "text"),
            ("verb", TfidfVectorizer(token_pattern=r"\S+", min_df=1, sublinear_tf=True), "verb_text"),
            ("num", StandardScaler(), [
                "lex_destructive", "lex_read", "n_params", "has_body", "ends_with_param", "path_depth",
            ]),
        ]
    )
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def hybrid_label(rule_lbl: str, rule_name: str, probs: dict[str, float],
                 threshold: float = DESTRUCTIVE_THRESHOLD) -> str:
    """Combine the high-precision rules with the model.

    1. safety first: a destructive probability above the threshold always wins
    2. a specific rule (explicit DELETE, destructive verb, read verb on POST) is trusted
    3. otherwise (the ambiguous default bucket) the model decides
    """
    if probs.get("destructive", 0.0) >= threshold:
        return "destructive"
    if rule_name in SPECIFIC_RULES:
        return rule_lbl
    return max(probs, key=probs.get)


def rule_label(method: str) -> str:
    m = method.upper()
    if m in ("GET", "HEAD", "OPTIONS"):
        return "read"
    if m == "DELETE":
        return "destructive"
    return "write"


@dataclass
class RiskPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    source: str  # model | rule

    @property
    def needs_review(self) -> bool:
        return self.source == "model" and self.confidence < LOW_CONFIDENCE


class RiskClassifier:
    def __init__(self, model_path: str | Path | None = None, destructive_threshold: float = DESTRUCTIVE_THRESHOLD):
        self.model = None
        self.destructive_threshold = destructive_threshold
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if path.exists():
            import joblib

            self.model = joblib.load(path)

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, ops: Iterable[Operation]) -> list[RiskPrediction]:
        ops = list(ops)
        if not ops:
            return []
        rules = [label_operation(o.method, o.path, o.operation_id, o.summary) for o in ops]
        if self.model is None:
            return [RiskPrediction(label, 1.0, {}, "rule") for label, _ in rules]
        proba = self.model.predict_proba(frame_from_operations(ops))
        classes = list(self.model.classes_)
        out = []
        for row, (rule_lbl, rule_name) in zip(proba, rules):
            probs = {c: float(p) for c, p in zip(classes, row)}
            label = hybrid_label(rule_lbl, rule_name, probs, self.destructive_threshold)
            out.append(RiskPrediction(label, probs[label], probs, "model"))
        return out

    def annotate(self, ops: list[Operation]) -> list[RiskPrediction]:
        preds = self.predict(ops)
        for op, pred in zip(ops, preds):
            # HTTP semantics are a hard floor: a DELETE is never downgraded by the model.
            if op.method.upper() == "DELETE" and pred.label != "destructive":
                pred = RiskPrediction("destructive", pred.confidence, pred.probabilities, pred.source + "+floor")
            op.risk = pred.label
            op.risk_confidence = round(pred.confidence, 3)
            if pred.needs_review:
                op.warnings.append(f"low-confidence risk label '{pred.label}' ({pred.confidence:.2f}); review")
        return preds


def annotation_hints(op: Operation) -> dict[str, bool]:
    """Map a risk label onto MCP ToolAnnotations hints."""
    method = op.method.upper()
    if op.risk == "read":
        return {"read_only_hint": True, "destructive_hint": False, "idempotent_hint": True, "open_world_hint": True}
    if op.risk == "destructive":
        return {"read_only_hint": False, "destructive_hint": True,
                "idempotent_hint": method in ("DELETE", "PUT"), "open_world_hint": True}
    return {"read_only_hint": False, "destructive_hint": False,
            "idempotent_hint": method in ("PUT",), "open_world_hint": True}
