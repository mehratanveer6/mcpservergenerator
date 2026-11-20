"""Feature engineering shared by training (SageMaker script mode) and inference.

Keeping this module inside the runtime package guarantees that the exact same
transformations run at training time and when the generator classifies a new
spec, which removes a common source of training/serving skew.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from .ir import Operation
from .lexicon import DESTRUCTIVE_VERBS, READ_VERBS

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")

FEATURE_COLUMNS = [
    "method", "text", "verb_text", "lex_destructive", "lex_read",
    "n_params", "has_body", "ends_with_param", "path_depth",
]


def split_identifier(text: str) -> str:
    """'PostChargesChargeRefund' / 'pulls/list-reviews' -> 'post charges charge refund' / 'pulls list reviews'."""
    text = _CAMEL.sub(" ", text or "")
    return _NON_ALNUM.sub(" ", text).strip().lower()


def path_words(path: str) -> str:
    parts = [p for p in (path or "").split("/") if p and not p.startswith("{")]
    return split_identifier(" ".join(parts))


def last_segment(path: str) -> str:
    parts = [p for p in (path or "").split("/") if p and not p.startswith("{")]
    return split_identifier(parts[-1]) if parts else ""


def row_from_fields(method: str, path: str, operation_id: str, summary: str, description: str,
                    tags: str, param_names: str, n_params: int, has_body: bool) -> dict:
    op_words = split_identifier(operation_id.split("/")[-1])
    summary_words = split_identifier(summary)
    text = " ".join([op_words, summary_words, split_identifier(description[:400]), path_words(path),
                     split_identifier(tags), split_identifier(param_names)])
    # Verb-position tokens are prefixed with the HTTP method so that a linear model can
    # learn interactions such as "post_search" (a read) versus "get_search".
    op_tokens, summary_tokens = op_words.split(), summary_words.split()
    verb_tokens = op_tokens[:2] + op_tokens[-1:] + summary_tokens[:2] + last_segment(path).split()
    m = (method or "GET").lower()
    verb_text = " ".join(f"{m}_{t}" for t in verb_tokens)
    stripped = (path or "").rstrip("/")
    return {
        "method": (method or "GET").upper(),
        "text": text,
        "verb_text": verb_text,
        "lex_destructive": int(any(t in DESTRUCTIVE_VERBS for t in verb_tokens)),
        "lex_read": int(any(t in READ_VERBS for t in verb_tokens)),
        "n_params": int(n_params),
        "has_body": int(bool(has_body)),
        "ends_with_param": int(stripped.endswith("}")),
        "path_depth": len([p for p in stripped.split("/") if p]),
    }


def frame_from_operations(ops: Iterable[Operation]) -> pd.DataFrame:
    rows = [
        row_from_fields(op.method, op.path, op.operation_id, op.summary, op.description,
                        " ".join(op.tags), " ".join(p.name for p in op.params), len(op.params),
                        op.body_schema is not None)
        for op in ops
    ]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def frame_from_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna("")
    rows = [
        row_from_fields(r.method, r.path, r.operation_id, r.summary, r.description, r.tags,
                        r.param_names, r.n_params, bool(r.has_body) and str(r.has_body) != "False")
        for r in df.itertuples(index=False)
    ]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
