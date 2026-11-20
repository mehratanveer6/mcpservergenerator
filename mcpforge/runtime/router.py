"""Tool router: find the operations that match a natural-language request.

This file is copied verbatim into every generated server, so it depends only on the
Python standard library. Two back ends are supported:

  * "bm25" (default): Okapi BM25 over tool name, summary, path and parameters
  * "embedding": cosine similarity over vectors precomputed at generation time
    (for example with Amazon Titan Text Embeddings V2 on Bedrock); the query
    vector is produced by a callable supplied by the server

Exposing a search tool instead of hundreds of tool definitions keeps the model's
context small, which is the retrieval-before-selection idea behind RAG-MCP.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Sequence

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an the of for to in on by with and or from at is are be this that it its as into "
    "my me i we our your you all any".split()
)
SYNONYMS = {
    "remove": "delete", "erase": "delete", "drop": "delete", "make": "create", "add": "create",
    "new": "create", "fetch": "get", "show": "get", "retrieve": "get", "find": "search",
    "lookup": "search", "edit": "update", "modify": "update", "change": "update",
    "pr": "pull", "prs": "pull", "repo": "repository", "repos": "repository", "issue": "issues",
}


def tokenize(text: str) -> list[str]:
    text = _CAMEL.sub(" ", text or "").lower()
    out = []
    for w in _WORD.findall(text):
        if w in STOPWORDS:
            continue
        w = SYNONYMS.get(w, w)
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # light plural folding
        out.append(w)
    return out


class BM25Router:
    def __init__(self, docs: Sequence[str], ids: Sequence[str], k1: float = 1.4, b: float = 0.75):
        self.ids = list(ids)
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.tf = [Counter(t) for t in self.tokens]
        self.lengths = [len(t) for t in self.tokens]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(term for toks in self.tokens for term in set(toks))
        n = len(self.tokens)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1))
            for term in q:
                f = tf.get(term)
                if f:
                    s += self.idf[term] * f * (self.k1 + 1) / (f + norm)
            scores.append(s)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.ids[i], round(scores[i], 4)) for i in order if scores[i] > 0]


class EmbeddingRouter:
    def __init__(self, vectors: Sequence[Sequence[float]], ids: Sequence[str], embed: Callable[[str], list[float]]):
        self.ids = list(ids)
        self.vectors = [self._unit(v) for v in vectors]
        self.embed = embed

    @staticmethod
    def _unit(v: Sequence[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        q = self._unit(self.embed(query))
        sims = [sum(a * b for a, b in zip(q, v)) for v in self.vectors]
        order = sorted(range(len(sims)), key=lambda i: -sims[i])[:k]
        return [(self.ids[i], round(sims[i], 4)) for i in order]


def document_for(entry: dict) -> str:
    """Text indexed for one catalog entry. Name and summary are repeated to weight them."""
    params = " ".join(p["name"] for p in entry.get("params", []))
    name = entry.get("tool_name", "").replace("_", " ")
    summary = entry.get("summary", "")
    return " ".join([name, name, summary, summary, entry.get("description", "")[:300],
                     entry.get("path", "").replace("/", " "), " ".join(entry.get("tags", [])), params])
