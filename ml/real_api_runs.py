"""Run the full generator on several real-world specs and collect the reports.

    python ml/real_api_runs.py spec1.yaml spec2.yaml ...
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcpforge.generator import GenerateOptions, generate  # noqa: E402


def main(paths: list[str]) -> None:
    rows = []
    for p in paths:
        out = Path(tempfile.mkdtemp(prefix="run_"))
        r = generate(p, GenerateOptions(output=out)).report
        rows.append({
            "api": r["api"], "format": r["format"], "operations": r["operations"], "mode": r["mode"],
            "direct_tools": r["direct_tools"], "risk_counts": r["risk_counts"],
            "spec_health": r["doctor"]["score"], "checks": [c["code"] for c in r["doctor"]["checks"]],
            "low_confidence": len(r["needs_review"]), "drift_status": r["drift"].get("status"),
            "drift": {k: r["drift"].get(k) for k in ("psi_method", "psi_text_length", "oov_rate", "mean_confidence")},
            "seconds": r["seconds"],
        })
        print(json.dumps(rows[-1]))
    (ROOT / "ml" / "reports" / "real_api_runs.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
