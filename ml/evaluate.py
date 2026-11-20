"""Model evaluation step for the SageMaker pipeline (also runnable locally).

Reads the trained model (model.tar.gz or a directory), scores it on the gold set
and writes evaluation.json, which the pipeline's ConditionStep reads to decide
whether the model is registered.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcpforge.features import frame_from_dataset  # noqa: E402
from mcpforge.risk import hybrid_label  # noqa: E402
from mcpforge.rules import label_operation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/opt/ml/processing/model")
    ap.add_argument("--data", default="/opt/ml/processing/data")
    ap.add_argument("--out", default="/opt/ml/processing/evaluation")
    a = ap.parse_args()

    model_path = Path(a.model)
    if (model_path / "model.tar.gz").exists():
        with tarfile.open(model_path / "model.tar.gz") as tar:
            tar.extractall(model_path, filter="data")
    model = joblib.load(model_path / "risk_model.joblib")

    ops = pd.read_csv(Path(a.data) / "operations.csv.gz").fillna("")
    gold = pd.read_csv(Path(a.data) / "gold_labels.csv").fillna("")
    rows = ops.loc[gold["row_id"]].reset_index(drop=True)
    proba = model.predict_proba(frame_from_dataset(rows))
    pred = np.array([
        hybrid_label(*label_operation(r.method, r.path, r.operation_id, r.summary), dict(zip(model.classes_, p)))
        for r, p in zip(rows.itertuples(index=False), proba)
    ])
    y = gold["gold_label"]
    report = {"metrics": {
        "macro_f1": {"value": float(f1_score(y, pred, average="macro"))},
        "accuracy": {"value": float(accuracy_score(y, pred))},
        "destructive_recall": {"value": float(recall_score(y, pred, labels=["destructive"], average="macro"))},
    }}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
