"""Curated mode: choose which operations become direct tools.

Converting every endpoint of a large API into a tool floods the model's context
and lowers selection accuracy. In curated mode at most `max_tools` operations are
exposed directly, chosen round-robin across tags so every functional area is
represented, and the remainder stays reachable through the search_operations and
call_operation meta-tools.
"""

from __future__ import annotations

from collections import defaultdict

from .ir import Operation

RISK_PRIORITY = {"read": 3.0, "write": 2.0, "destructive": 1.0, "unknown": 1.5}


def priority(op: Operation) -> float:
    score = RISK_PRIORITY.get(op.risk, 1.0)
    if op.summary or op.description:
        score += 1.0
    depth = len([p for p in op.path.split("/") if p])
    score -= 0.15 * depth
    if op.method == "GET" and not op.path.rstrip("/").endswith("}"):
        score += 0.5  # collection listings are good entry points
    if op.method == "GET" and op.path.rstrip("/").endswith("}"):
        score += 0.3  # fetch-by-id is the usual follow-up
    if op.warnings:
        score -= 0.5
    return score


def group_key(op: Operation) -> str:
    if op.tags:
        return op.tags[0]
    parts = [p for p in op.path.split("/") if p and not p.startswith("{") and not p.startswith("v")]
    return parts[0] if parts else "default"


def select(ops: list[Operation], max_tools: int, allowed_risks: set[str]) -> list[Operation]:
    eligible = [o for o in ops if not o.deprecated and o.risk in allowed_risks]
    if len(eligible) <= max_tools:
        return eligible
    groups: dict[str, list[Operation]] = defaultdict(list)
    for op in eligible:
        groups[group_key(op)].append(op)
    queues = sorted(
        (sorted(g, key=priority, reverse=True) for g in groups.values()),
        key=lambda g: -priority(g[0]),
    )
    chosen: list[Operation] = []
    while len(chosen) < max_tools and any(queues):
        for q in queues:
            if q and len(chosen) < max_tools:
                chosen.append(q.pop(0))
    order = {id(o): i for i, o in enumerate(ops)}
    return sorted(chosen, key=lambda o: order[id(o)])
