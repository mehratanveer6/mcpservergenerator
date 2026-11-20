"""Train the operation risk classifier.

Runs unchanged locally or as an Amazon SageMaker training job in script mode:
SageMaker passes hyperparameters as CLI arguments and exposes the data channels
and model directory through SM_* environment variables.

    python ml/train.py --train ml/data --validation ml/data --C 8 --ngram-max 1

The final line printed is parsed by SageMaker Automatic Model Tuning through the
metric regex  'validation:macro_f1=([0-9\\.]+)'.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcpforge.features import frame_from_dataset  # noqa: E402
from mcpforge.risk import build_pipeline  # noqa: E402

HELD_OUT_PROVIDERS = [
    "stripe.com", "slack.com", "squareup.com", "plaid.com",
    "spotify.com", "asana.com", "mux.com", "openai.com",
]


def load_training_frame(channel: str) -> pd.DataFrame:
    path = Path(channel)
    files = sorted(path.glob("operations*.csv*")) if path.is_dir() else [path]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).fillna("")
    return df[~df["provider"].isin(HELD_OUT_PROVIDERS)].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=8.0)
    ap.add_argument("--max-features", type=int, default=5000)
    ap.add_argument("--ngram-max", type=int, default=1)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", str(ROOT / "ml" / "data")))
    ap.add_argument("--validation", default=os.environ.get("SM_CHANNEL_VALIDATION", str(ROOT / "ml" / "data")))
    ap.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", str(ROOT / "mcpforge" / "models")))
    args = ap.parse_args()

    train_df = load_training_frame(args.train)
    X, y = frame_from_dataset(train_df), train_df["weak_label"]
    model = build_pipeline(C=args.C, max_features=args.max_features, ngram_max=args.ngram_max, min_df=args.min_df)
    model.fit(X, y)
    print(f"trained on {len(train_df)} operations from {train_df['provider'].nunique()} providers")

    gold_path = Path(args.validation) / "gold_labels.csv"
    macro_f1 = float("nan")
    if gold_path.exists():
        gold = pd.read_csv(gold_path).fillna("")
        full = pd.concat([pd.read_csv(f) for f in sorted(Path(args.train).glob("operations*.csv*"))]).fillna("")
        rows = full.loc[gold["row_id"]]
        pred = model.predict(frame_from_dataset(rows))
        macro_f1 = f1_score(gold["gold_label"], pred, average="macro")
        print(classification_report(gold["gold_label"], pred, digits=3))

    out = Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "risk_model.joblib", compress=3)
    (out / "hyperparameters.json").write_text(json.dumps(vars(args), indent=2))
    print(f"validation:macro_f1={macro_f1:.4f};")


if __name__ == "__main__":
    main()
