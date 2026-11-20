"""Tool naming and description synthesis.

Good tool descriptions state what the tool does, when to use it and what it
changes. Descriptions are built deterministically from the spec; when the
--enhance bedrock option is used, a foundation model on Amazon Bedrock rewrites
them and the rewrite is accepted only if it passes the poisoning scanner.
"""

from __future__ import annotations

import keyword
import re

from .ir import ApiSpec, Operation
from .poisoning import sanitize, scan

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_BAD = re.compile(r"[^a-z0-9_]+")
MAX_TOOL_NAME = 64
RISK_NOTE = {
    "read": "Read-only; does not change data.",
    "write": "Changes data on the remote service.",
    "destructive": "DESTRUCTIVE: deletes, revokes, cancels or reverses data. Confirm with the user first.",
}


def snake(text: str) -> str:
    text = _CAMEL.sub("_", text or "").lower()
    return _BAD.sub("_", text).strip("_")


def python_identifier(name: str) -> str:
    ident = snake(name) or "arg"
    if ident[0].isdigit():
        ident = "p_" + ident
    if keyword.iskeyword(ident) or ident in {"self", "body", "ctx"}:
        ident += "_"
    return ident


def tool_name(op: Operation, used: set[str]) -> str:
    base = snake(op.operation_id.replace("/", "_")) or snake(f"{op.method}_{op.path}")
    base = re.sub(r"_+", "_", base)[:MAX_TOOL_NAME].strip("_") or "operation"
    name, n = base, 2
    while name in used:
        suffix = f"_{n}"
        name = base[: MAX_TOOL_NAME - len(suffix)] + suffix
        n += 1
    used.add(name)
    return name


def first_sentence(text: str, limit: int = 300) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    m = re.match(r"(.{20,}?[.!?])(\s|$)", text)
    out = m.group(1) if m else text
    return out[:limit]


def humanize_operation_id(op_id: str) -> str:
    words = snake(op_id.split("/")[-1]).replace("_", " ").strip()
    return words[:1].upper() + words[1:] if words else ""


WITHHELD = "Original description withheld by MCPForge: it contained instruction-like text (possible tool poisoning)."


def _suspicious(text: str) -> bool:
    return any(f.severity in ("high", "medium") for f in scan(text))


def build_description(op: Operation) -> str:
    if _suspicious(op.summary) or _suspicious(op.description):
        lead = humanize_operation_id(op.operation_id) or f"{op.method} {op.path}"
        return " ".join([lead.rstrip(".") + ".", WITHHELD, RISK_NOTE.get(op.risk, ""), f"HTTP {op.method} {op.path}."])
    summary = sanitize(op.summary, 200)
    detail = sanitize(first_sentence(op.description), 300)
    lead = summary or detail or humanize_operation_id(op.operation_id)
    parts = [lead.rstrip(".") + "."]
    if detail and detail.lower() != summary.lower() and not detail.lower().startswith(summary.lower()[:30] or "~"):
        parts.append(detail)
    parts.append(RISK_NOTE.get(op.risk, ""))
    parts.append(f"HTTP {op.method} {op.path}.")
    return " ".join(p for p in parts if p)


def assign_names_and_descriptions(spec: ApiSpec) -> None:
    used: set[str] = set()
    for op in spec.operations:
        op.tool_name = tool_name(op, used)
        op.tool_description = build_description(op)


def enhance_with_bedrock(spec: ApiSpec, model_id: str, region: str | None = None, limit: int = 60) -> int:
    """Rewrite descriptions with a Bedrock model; returns how many were accepted."""
    from .aws.bedrock import rewrite_description

    accepted = 0
    for op in spec.operations[:limit]:
        try:
            text = rewrite_description(op, model_id=model_id, region=region)
        except Exception as exc:  # noqa: BLE001 - enhancement is best-effort
            op.warnings.append(f"bedrock rewrite failed: {type(exc).__name__}")
            continue
        if text and not scan(text):
            op.tool_description = sanitize(text, 600) + " " + RISK_NOTE.get(op.risk, "")
            accepted += 1
    return accepted
