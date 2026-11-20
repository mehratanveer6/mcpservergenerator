"""Measure the tool-poisoning scanner.

Positives: 20 attack descriptions in ml/data/poisoning_cases.csv covering the
categories reported in the MCP security literature (hidden instruction blocks,
instruction override, concealment, secret-file access, exfiltration, tool-chaining
coercion, invisible Unicode).
Negatives: every operation (summary, description, parameter descriptions) of the
real-world specs passed on the command line.

    python ml/poisoning_eval.py spec1.yaml spec2.yaml ...
"""

from __future__ import annotations

import codecs
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcpforge.loader import load_spec  # noqa: E402
from mcpforge.describe import build_description  # noqa: E402
from mcpforge.ir import Operation  # noqa: E402
from mcpforge.poisoning import scan  # noqa: E402


def flagged(text: str, level: str) -> bool:
    severities = {"high"} if level == "high" else {"high", "medium"}
    return any(f.severity in severities for f in scan(text))


def main(specs: list[str]) -> None:
    cases = pd.read_csv(ROOT / "ml" / "data" / "poisoning_cases.csv")
    cases["text"] = [codecs.decode(t, "unicode_escape") if "\\u" in t else t for t in cases["text"]]
    benign = []
    for s in specs:
        for op in load_spec(s).operations:
            benign.append(" ".join([op.summary, op.description] + [p.description for p in op.params]))
    out = {"positives": len(cases), "negatives": len(benign)}
    for level in ("high", "high+medium"):
        tp = sum(flagged(t, level) for t in cases["text"])
        fp = sum(flagged(t, level) for t in benign)
        out[level] = {"detected": tp, "recall": round(tp / len(cases), 3), "false_positives": fp,
                      "false_positive_rate": round(fp / max(len(benign), 1), 4)}
    per_cat = {}
    for cat, g in cases.groupby("category"):
        per_cat[cat] = f"{sum(flagged(t, 'high+medium') for t in g['text'])}/{len(g)}"
    out["by_category"] = per_cat
    # what actually reaches the model: descriptions written into generated servers
    generated = [build_description(Operation("op", "GET", "/x", summary="", description=t)) for t in cases["text"]]
    out["generated_descriptions_still_flagged"] = sum(flagged(t, "high+medium") for t in generated)
    (ROOT / "ml" / "reports" / "poisoning_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
