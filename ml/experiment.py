"""Full offline experiment for the risk classifier.

Steps
 1. provider-grouped split: 8 providers are held out completely
 2. baselines: HTTP-method rule and the weak labelling functions
 3. hyperparameter search (GroupKFold by provider) for logistic regression,
    the same search space used by the SageMaker tuning job in aws/
 4. comparison against LinearSVC, Complement Naive Bayes and Random Forest
 5. evaluation on the manually reviewed gold set (held-out providers only)
 6. artefacts: model, metrics.json, confusion matrix, drift baseline

    python ml/experiment.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, classification_report, confusion_matrix, f1_score, recall_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold  # noqa: E402
from sklearn.naive_bayes import ComplementNB  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from mcpforge.features import frame_from_dataset  # noqa: E402
from mcpforge.risk import DESTRUCTIVE_THRESHOLD, build_pipeline, hybrid_label, rule_label  # noqa: E402
from mcpforge.rules import label_operation  # noqa: E402
from mcpforge.drift import build_baseline  # noqa: E402
from train import HELD_OUT_PROVIDERS  # noqa: E402

DATA = ROOT / "ml" / "data"
REPORTS = ROOT / "ml" / "reports"
LABELS = ["read", "write", "destructive"]


def scores(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "destructive_recall": round(recall_score(y_true, y_pred, labels=["destructive"], average="macro"), 4),
    }


def with_threshold(model, X, threshold=DESTRUCTIVE_THRESHOLD):
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    pred = np.array(classes)[proba.argmax(axis=1)]
    pred[proba[:, classes.index("destructive")] >= threshold] = "destructive"
    return pred


def swap_classifier(pipe, clf):
    from sklearn.base import clone

    p = clone(pipe)
    p.steps[-1] = ("clf", clf)
    return p


def main() -> None:
    t0 = time.time()
    REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "operations.csv.gz").fillna("")
    gold = pd.read_csv(DATA / "gold_labels.csv").fillna("")
    gold_rows = df.loc[gold["row_id"]].reset_index(drop=True)
    y_gold = gold["gold_label"].to_numpy()
    X_gold = frame_from_dataset(gold_rows)
    post_mask = (gold_rows["method"].isin(["POST", "PUT", "PATCH"])).to_numpy()

    train = df[~df["provider"].isin(HELD_OUT_PROVIDERS)].reset_index(drop=True)
    test = df[df["provider"].isin(HELD_OUT_PROVIDERS)].reset_index(drop=True)
    X_train, y_train, groups = frame_from_dataset(train), train["weak_label"], train["provider"]
    X_test = frame_from_dataset(test)

    results: dict = {
        "dataset": {
            "operations": int(len(df)), "apis": int(df["api"].nunique()), "providers": int(df["provider"].nunique()),
            "train_operations": int(len(train)), "train_providers": int(train["provider"].nunique()),
            "heldout_operations": int(len(test)), "heldout_providers": HELD_OUT_PROVIDERS,
            "gold_operations": int(len(gold)), "weak_label_distribution": df["weak_label"].value_counts().to_dict(),
            "gold_label_distribution": gold["gold_label"].value_counts().to_dict(),
            "method_distribution": df["method"].value_counts().to_dict(),
        },
        "gold": {}, "gold_mutating_only": {}, "heldout_weak_agreement": {},
    }

    # --- baselines -------------------------------------------------------------------
    method_pred = np.array([rule_label(m) for m in gold_rows["method"]])
    weak_pred = gold_rows["weak_label"].to_numpy()
    results["gold"]["method_rule"] = scores(y_gold, method_pred)
    results["gold"]["weak_labels"] = scores(y_gold, weak_pred)
    results["gold_mutating_only"]["method_rule"] = scores(y_gold[post_mask], method_pred[post_mask])
    results["gold_mutating_only"]["weak_labels"] = scores(y_gold[post_mask], weak_pred[post_mask])

    # --- hyperparameter search ---------------------------------------------------------
    grid = {
        "clf__C": [0.5, 1.0, 2.0, 4.0, 8.0],
        "features__text__ngram_range": [(1, 1), (1, 2)],
        "features__text__max_features": [5000, 20000],
    }
    search = GridSearchCV(build_pipeline(), grid, scoring="f1_macro", cv=GroupKFold(n_splits=5), n_jobs=-1)
    search.fit(X_train, y_train, groups=groups)
    best = search.best_estimator_
    cv_table = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    cv_table.sort_values("rank_test_score").head(10).to_csv(REPORTS / "hpo_top10.csv", index=False)
    results["hpo"] = {
        "search_space": {k: [str(v) for v in vs] for k, vs in grid.items()},
        "candidates": int(len(cv_table)), "folds": 5,
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "best_cv_macro_f1": round(float(search.best_score_), 4),
    }

    # --- model comparison ------------------------------------------------------------------
    candidates = {
        "logreg_tuned": best,
        "linear_svm": swap_classifier(best, CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3)),
        "complement_nb": swap_classifier(best, ComplementNB()),
        "random_forest": swap_classifier(best, RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                                      n_jobs=-1, random_state=7)),
    }
    for name, model in candidates.items():
        if name != "logreg_tuned":
            if name == "complement_nb":
                # ComplementNB needs non-negative inputs: drop the scaled numeric block
                model.named_steps["features"].transformers = [
                    t for t in model.named_steps["features"].transformers if t[0] != "num"
                ]
            model.fit(X_train, y_train)
        pred = model.predict(X_gold)
        results["gold"][name] = scores(y_gold, pred)
        results["gold_mutating_only"][name] = scores(y_gold[post_mask], pred[post_mask])
        results["heldout_weak_agreement"][name] = scores(test["weak_label"], model.predict(X_test))

    thr_pred = with_threshold(best, X_gold)
    results["gold"]["logreg_tuned+safety_threshold"] = scores(y_gold, thr_pred)
    results["gold_mutating_only"]["logreg_tuned+safety_threshold"] = scores(y_gold[post_mask], thr_pred[post_mask])
    # hybrid: high-precision rules + model for the ambiguous bucket + safety threshold
    proba = best.predict_proba(X_gold)
    classes = list(best.classes_)
    hybrid_pred = np.array([
        hybrid_label(*label_operation(r.method, r.path, r.operation_id, r.summary), dict(zip(classes, p)))
        for r, p in zip(gold_rows.itertuples(index=False), proba)
    ])
    results["gold"]["hybrid_final"] = scores(y_gold, hybrid_pred)
    results["gold_mutating_only"]["hybrid_final"] = scores(y_gold[post_mask], hybrid_pred[post_mask])
    for thr in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        tp = np.array([
            hybrid_label(*label_operation(r.method, r.path, r.operation_id, r.summary), dict(zip(classes, p)), thr)
            for r, p in zip(gold_rows.itertuples(index=False), proba)
        ])
        results.setdefault("threshold_sweep", {})[str(thr)] = scores(y_gold, tp)
    thr_pred = hybrid_pred
    results["final_model_report"] = classification_report(y_gold, thr_pred, labels=LABELS, output_dict=True)

    # disagreements between the final model and the weak labels, useful for the report
    diff = gold_rows.assign(gold=y_gold, model=thr_pred)
    diff = diff[(diff["model"] != diff["gold"]) | (diff["weak_label"] != diff["gold"])][["method", "path", "operation_id", "weak_label", "model", "gold"]]
    diff.to_csv(REPORTS / "model_vs_rules_disagreements.csv", index=False)

    # --- confusion matrix ------------------------------------------------------------------
    cm = confusion_matrix(y_gold, thr_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=200)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold label")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Hybrid risk classifier (gold set)")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png")
    results["confusion_matrix"] = {"labels": LABELS, "matrix": cm.tolist()}

    # --- artefacts ------------------------------------------------------------------------------
    joblib.dump(best, ROOT / "mcpforge" / "models" / "risk_model.joblib", compress=3)
    baseline = build_baseline(X_train, best)
    (ROOT / "mcpforge" / "models" / "drift_baseline.json").write_text(json.dumps(baseline, indent=2))
    results["runtime_seconds"] = round(time.time() - t0, 1)
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps({k: results[k] for k in ("hpo", "gold", "gold_mutating_only", "threshold_sweep")}, indent=2))


if __name__ == "__main__":
    main()
