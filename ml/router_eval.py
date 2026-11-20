"""Evaluate tool routing and context cost.

Part A, retrieval: 65 hand-written natural-language requests against the full
operation catalogs of GitHub (845 operations), Stripe (452) and Slack (174).
Metrics: Recall@1, Recall@5, Recall@10 and mean reciprocal rank (MRR).

Part B, context cost (optional, needs the spec files): size of the tools/list
response a client must load, for all-tools mode versus curated mode.

    python ml/router_eval.py [--specs github.yaml stripe.yaml slack.yaml]
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcpforge.describe import snake  # noqa: E402
from mcpforge.runtime.router import BM25Router, document_for  # noqa: E402

DATA = ROOT / "ml" / "data"
REPORTS = ROOT / "ml" / "reports"
API_NAMES = {"github.com": "github.com/api.github.com", "stripe.com": "stripe.com", "slack.com": "slack.com"}


def catalog_from_dataset(df: pd.DataFrame) -> list[dict]:
    return [{
        "operation_id": r.operation_id, "tool_name": snake(r.operation_id.replace("/", "_")),
        "summary": r.summary, "description": r.description, "path": r.path, "tags": r.tags.split(),
        "params": [{"name": n} for n in r.param_names.split()],
    } for r in df.itertuples(index=False)]


def rank_metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    return {
        "recall@1": round(sum(r is not None and r < 1 for r in ranks) / n, 3),
        "recall@5": round(sum(r is not None and r < 5 for r in ranks) / n, 3),
        "recall@10": round(sum(r is not None and r < 10 for r in ranks) / n, 3),
        "mrr": round(sum(1 / (r + 1) for r in ranks if r is not None) / n, 3),
    }


def retrieval() -> dict:
    df = pd.read_csv(DATA / "operations.csv.gz").fillna("")
    queries = pd.read_csv(DATA / "router_queries.csv")
    out: dict = {}
    all_bm25, all_tfidf, all_name = [], [], []
    for provider, q in queries.groupby("provider"):
        ops = df[df["api"] == API_NAMES[provider]]
        entries = catalog_from_dataset(ops.drop_duplicates("operation_id"))
        ids = [e["operation_id"] for e in entries]
        docs = [document_for(e) for e in entries]
        bm25 = BM25Router(docs, ids)
        vec = TfidfVectorizer(sublinear_tf=True).fit(docs)
        D = vec.transform(docs)
        name_vec = TfidfVectorizer().fit([e["tool_name"].replace("_", " ") for e in entries])
        N = name_vec.transform([e["tool_name"].replace("_", " ") for e in entries])
        r_bm25, r_tfidf, r_name = [], [], []
        for query, expected in zip(q["query"], q["expected"]):
            hits = [h for h, _ in bm25.search(query, k=len(ids))]
            r_bm25.append(hits.index(expected) if expected in hits else None)
            order = list(np.argsort(-cosine_similarity(vec.transform([query]), D)[0]))
            r_tfidf.append(order.index(ids.index(expected)))
            order = list(np.argsort(-cosine_similarity(name_vec.transform([query]), N)[0]))
            r_name.append(order.index(ids.index(expected)))
        out[provider] = {"operations": len(ids), "queries": len(q),
                         "bm25_router": rank_metrics(r_bm25), "tfidf_cosine": rank_metrics(r_tfidf),
                         "tool_name_only": rank_metrics(r_name),
                         "random_recall@5": round(5 / len(ids), 4)}
        all_bm25 += r_bm25
        all_tfidf += r_tfidf
        all_name += r_name
    out["overall"] = {"queries": len(queries), "bm25_router": rank_metrics(all_bm25),
                      "tfidf_cosine": rank_metrics(all_tfidf), "tool_name_only": rank_metrics(all_name)}
    return out


async def _tool_list_size(server_dir: Path) -> tuple[int, int]:
    from mcp import Client

    os.environ["MCPFORGE_POLICY"] = "full"
    spec = importlib.util.spec_from_file_location(f"gen_{server_dir.name}", server_dir / "server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    async with Client(module.mcp) as client:
        tools = (await client.list_tools()).tools
    payload = json.dumps([t.model_dump(mode="json", exclude_none=True, by_alias=True) for t in tools])
    return len(tools), len(payload)


def context_cost(spec_paths: list[str]) -> dict:
    from mcpforge.generator import GenerateOptions, generate

    out = {}
    for path in spec_paths:
        row = {}
        for mode in ("all", "curated"):
            d = Path(tempfile.mkdtemp(prefix=f"eval_{mode}_"))
            res = generate(path, GenerateOptions(output=d, mode=mode, max_tools=25, policy="full"))
            n_tools, chars = asyncio.run(_tool_list_size(d))
            row[mode] = {"tools": n_tools, "chars": chars, "approx_tokens": chars // 4}
        row["reduction"] = round(1 - row["curated"]["chars"] / row["all"]["chars"], 3)
        out[res.spec.title] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="*", default=[])
    args = ap.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    results = {"retrieval": retrieval()}
    if args.specs:
        results["context_cost"] = context_cost(args.specs)
    (REPORTS / "router_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
