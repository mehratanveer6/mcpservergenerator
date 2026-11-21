"""Build the static demo site in docs/ from real generator runs.

    python tools/build_site.py examples/petclinic.yaml examples/poisoned.yaml github.yaml stripe.yaml slack.yaml

Each spec is run through the full generator. The page shows the reports, the risk labels,
the generated files, and offers the generated server as a zip download.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcpforge.generator import GenerateOptions, generate  # noqa: E402
from mcpforge.poisoning import scan  # noqa: E402

DOCS = ROOT / "docs"
SHOW_FILES = ["server.py", "catalog.json", "mcpforge_report.json", "README.md", "Dockerfile", "pyproject.toml", ".env.example"]
MAX_FILE_CHARS = 60_000


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def run(spec_path: str) -> dict:
    out = Path(tempfile.mkdtemp(prefix="site_"))
    res = generate(spec_path, GenerateOptions(output=out))
    r = res.report
    cat = json.loads((out / "catalog.json").read_text())
    key = slug(r["api"])[:40]
    review = {x.get("operation_id") if isinstance(x, dict) else x for x in r["needs_review"]}
    ops = [{"m": o["method"], "p": o["path"], "t": o["tool_name"], "r": o["risk"], "c": round(o["risk_confidence"], 3),
            "d": bool(o.get("direct")), "s": (o.get("summary") or "")[:140], "v": o["operation_id"] in review}
           for o in cat["operations"]]
    findings = []
    for op in res.spec.operations:
        for f in scan(op.summary + " " + op.description) + [f for p in op.params for f in scan(p.description)]:
            findings.append({"op": op.operation_id, "rule": f.rule, "severity": f.severity, "excerpt": f.excerpt[:160]})
    files = {}
    for name in SHOW_FILES:
        p = out / name
        if p.exists():
            text = p.read_text()
            if len(text) > MAX_FILE_CHARS:
                text = text[:MAX_FILE_CHARS] + f"\n\n... ({len(text) - MAX_FILE_CHARS:,} more characters in the downloadable project)\n"
            files[name] = text
    (DOCS / "downloads").mkdir(parents=True, exist_ok=True)
    zpath = DOCS / "downloads" / f"{key}-mcp-server.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                z.write(p, f"{key}-mcp-server/{p.relative_to(out)}")
    return {"key": key, "api": r["api"], "format": r["format"], "operations": r["operations"], "mode": r["mode"],
            "direct_tools": r["direct_tools"], "policy": r["policy"], "risk_counts": r["risk_counts"],
            "classifier": r["classifier"], "needs_review": len(r["needs_review"]), "doctor": r["doctor"],
            "drift": r["drift"], "warnings": r["warnings"], "seconds": r["seconds"], "base_url": cat.get("base_url"),
            "ops": ops, "findings": findings, "files": files, "zip": f"downloads/{zpath.name}",
            "zip_kb": round(zpath.stat().st_size / 1024)}


def main(paths: list[str]) -> None:
    rep = ROOT / "ml" / "reports"
    data = {"runs": [run(p) for p in paths],
            "metrics": json.loads((rep / "metrics.json").read_text()),
            "router": json.loads((rep / "router_metrics.json").read_text()),
            "poisoning": json.loads((rep / "poisoning_metrics.json").read_text())}
    data["metrics"].pop("threshold_sweep", None)
    (DOCS / "data.js").write_text("window.MCPFORGE = " + json.dumps(data, separators=(",", ":")) + ";\n")
    for r in data["runs"]:
        print(f"{r['api']}: {r['operations']} ops, {r['direct_tools']} direct, health {r['doctor']['score']}, zip {r['zip_kb']} KB")


if __name__ == "__main__":
    main(sys.argv[1:])
