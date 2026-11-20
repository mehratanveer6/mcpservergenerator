"""Build the operation dataset from a directory of real-world OpenAPI documents.

Usage:
    python ml/build_dataset.py --specs /path/to/openapi-directory/APIs --out ml/data/operations.csv.gz

The input layout follows the public APIs.guru openapi-directory
(APIs/<provider>[/<service>]/<version>/openapi.yaml). Only the newest version of
each provider/service is kept so that near-duplicate versions do not leak between
train and test splits.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcpforge.loader import SpecError, load_spec  # noqa: E402
from mcpforge.rules import label_operation  # noqa: E402


def newest_versions(root: Path) -> list[Path]:
    files = sorted(list(root.rglob("openapi.yaml")) + list(root.rglob("swagger.yaml")))
    groups: dict[Path, Path] = {}
    for f in files:
        key = f.parent.parent  # provider[/service]
        groups[key] = f  # sorted order -> last one wins (newest version string)
    return sorted(groups.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--out", default="ml/data/operations.csv.gz")
    args = ap.parse_args()

    root = Path(args.specs)
    rows = []
    failures = 0
    t0 = time.time()
    for f in newest_versions(root):
        rel = f.relative_to(root)
        provider = rel.parts[0]
        api_name = "/".join(rel.parts[:-2])
        try:
            spec = load_spec(f, max_ref_depth=2)
        except (SpecError, Exception) as exc:  # noqa: BLE001 - a corpus always has a few broken files
            failures += 1
            print(f"skip {rel}: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        for op in spec.operations:
            label, rule = label_operation(op.method, op.path, op.operation_id, op.summary)
            rows.append(
                {
                    "provider": provider,
                    "api": api_name,
                    "operation_id": op.operation_id,
                    "method": op.method,
                    "path": op.path,
                    "summary": op.summary[:300],
                    "description": op.description[:600].replace("\n", " "),
                    "tags": " ".join(op.tags),
                    "param_names": " ".join(p.name for p in op.params),
                    "n_params": len(op.params),
                    "has_body": op.body_schema is not None,
                    "synthesized_id": op.synthesized_id,
                    "weak_label": label,
                    "weak_rule": rule,
                }
            )
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, compression="gzip")
    print(
        f"{len(df)} operations from {df['api'].nunique()} APIs ({df['provider'].nunique()} providers); "
        f"{failures} files skipped; {time.time() - t0:.1f}s"
    )
    print(df["weak_label"].value_counts().to_string())
    print(df["weak_rule"].value_counts().to_string())


if __name__ == "__main__":
    main()
