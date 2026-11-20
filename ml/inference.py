"""SageMaker inference handlers for the risk classifier (scikit-learn container).

Request  (application/json): {"operations": [{"method": "POST", "path": "/v1/refunds",
                              "operation_id": "PostRefunds", "summary": "Create a refund"}, ...]}
Response (application/json): {"predictions": [{"label": "destructive", "probabilities": {...}}, ...]}
"""

from __future__ import annotations

import json
import os

import joblib

from mcpforge.features import row_from_fields
from mcpforge.risk import hybrid_label
from mcpforge.rules import label_operation

import pandas as pd


def model_fn(model_dir: str):
    return joblib.load(os.path.join(model_dir, "risk_model.joblib"))


def input_fn(body, content_type: str = "application/json"):
    if content_type != "application/json":
        raise ValueError(f"unsupported content type {content_type}")
    payload = json.loads(body)
    return payload.get("operations", payload if isinstance(payload, list) else [])


def predict_fn(operations: list[dict], model):
    rows = [
        row_from_fields(o.get("method", "GET"), o.get("path", ""), o.get("operation_id", ""),
                        o.get("summary", ""), o.get("description", ""), " ".join(o.get("tags", [])),
                        " ".join(o.get("param_names", [])), int(o.get("n_params", 0)), bool(o.get("has_body", False)))
        for o in operations
    ]
    proba = model.predict_proba(pd.DataFrame(rows))
    out = []
    for o, p in zip(operations, proba):
        probs = {c: round(float(v), 4) for c, v in zip(model.classes_, p)}
        rule = label_operation(o.get("method", "GET"), o.get("path", ""), o.get("operation_id", ""), o.get("summary", ""))
        out.append({"label": hybrid_label(*rule, probs), "probabilities": probs, "rule": rule[1]})
    return out


def output_fn(predictions, accept: str = "application/json"):
    return json.dumps({"predictions": predictions}), "application/json"
