"""Transparent rule-based risk labelling (labelling functions).

Used in two places:
  * ml/build_dataset.py, to weakly label the training corpus
  * mcpforge.risk, as the high-precision half of the hybrid classifier
"""

from __future__ import annotations

import re

from .lexicon import DESTRUCTIVE_VERBS, READ_VERBS

_TOKEN = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?![a-z])")
SPECIFIC_RULES = frozenset({"lf_http_delete", "lf_destructive_verb", "lf_post_read_verb", "lf_post_read_segment"})


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def leading_verb(op_id: str, summary: str) -> str:
    candidates = []
    summary_tokens = tokens(summary)
    if summary_tokens:
        candidates.append(summary_tokens[0])
    id_tokens = tokens(op_id.split("/")[-1])  # "pulls/list" -> "list"
    if id_tokens:
        candidates.append(id_tokens[0])
    for c in candidates:
        if c in DESTRUCTIVE_VERBS or c in READ_VERBS:
            return c
    return candidates[0] if candidates else ""


def label_operation(method: str, path: str, op_id: str, summary: str) -> tuple[str, str]:
    """Return (label, rule_name) for one operation."""
    method = method.upper()
    last = path.rstrip("/").split("/")[-1].lower() if path else ""
    verb = leading_verb(op_id, summary)
    seg_tokens = set(tokens(last)) if not last.startswith("{") else set()

    if method == "DELETE":
        return "destructive", "lf_http_delete"
    if method in ("GET", "HEAD", "OPTIONS"):
        return "read", "lf_http_safe_method"
    if verb in DESTRUCTIVE_VERBS or seg_tokens & DESTRUCTIVE_VERBS:
        return "destructive", "lf_destructive_verb"
    if method == "POST" and verb in READ_VERBS:
        return "read", "lf_post_read_verb"
    if method == "POST" and seg_tokens & {"search", "query", "lookup", "preview", "validate", "estimate"}:
        return "read", "lf_post_read_segment"
    return "write", "lf_default_write"
