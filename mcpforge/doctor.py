"""Spec doctor: data-quality checks run before any code is generated.

The checks target the defect classes that most often break generated servers in
practice (missing or relative base URLs, missing security schemes, missing
operation ids) and the description problems that hurt tool selection by a model
(missing summaries, undocumented parameters, duplicate names, very large tool
counts). Each finding carries a severity; the overall health score starts at 100
and is reduced per finding, weighted by how many operations are affected.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

from .ir import ApiSpec
from .poisoning import scan

WEIGHTS = {"error": 25, "warning": 8, "info": 2}


@dataclass
class Check:
    code: str
    severity: str  # error | warning | info
    message: str
    affected: int = 0


@dataclass
class DoctorReport:
    score: int
    checks: list[Check] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"score": self.score, "stats": self.stats, "checks": [asdict(c) for c in self.checks]}


def diagnose(spec: ApiSpec, curated_threshold: int = 40) -> DoctorReport:
    checks: list[Check] = []
    ops = spec.operations
    n = len(ops)

    if not spec.base_url:
        checks.append(Check("NO_BASE_URL", "error", "No server URL in spec; set API_BASE_URL at runtime."))
    elif not spec.base_url.startswith(("http://", "https://")):
        checks.append(Check("RELATIVE_BASE_URL", "error",
                            f"Server URL '{spec.raw_server_url}' is relative; set API_BASE_URL at runtime."))
    if not spec.security_schemes:
        checks.append(Check("NO_SECURITY_SCHEMES", "warning",
                            "No security schemes declared; authentication must be configured manually."))

    unsupported = [k for k, v in spec.security_schemes.items()
                   if isinstance(v, dict) and v.get("type") in ("openIdConnect", "mutualTLS")]
    if unsupported:
        checks.append(Check("UNSUPPORTED_AUTH", "warning",
                            f"Auth schemes {unsupported} need a manual token; a bearer token env var is generated."))

    synthesized = sum(o.synthesized_id for o in ops)
    if synthesized:
        checks.append(Check("MISSING_OPERATION_ID", "warning",
                            "Operations without operationId; tool names were synthesised from method and path.",
                            synthesized))
    dup_paths = [k for k, c in Counter((o.method, o.path) for o in ops).items() if c > 1]
    if dup_paths:
        checks.append(Check("DUPLICATE_OPERATION", "warning", "Duplicate method and path pairs.", len(dup_paths)))

    no_doc = sum(1 for o in ops if not o.summary and not o.description)
    if no_doc:
        checks.append(Check("MISSING_DESCRIPTION", "warning",
                            "Operations with neither summary nor description; the model must guess their purpose.",
                            no_doc))
    params = [p for o in ops for p in o.params]
    undocumented = sum(1 for p in params if not p.description)
    if params and undocumented / len(params) > 0.3:
        checks.append(Check("UNDOCUMENTED_PARAMETERS", "info",
                            f"{undocumented}/{len(params)} parameters have no description.", undocumented))
    long_desc = sum(1 for o in ops if len(o.description) > 2000)
    if long_desc:
        checks.append(Check("LONG_DESCRIPTION", "info", "Very long descriptions will be truncated.", long_desc))
    deprecated = sum(o.deprecated for o in ops)
    if deprecated:
        checks.append(Check("DEPRECATED_OPERATIONS", "info", "Deprecated operations are excluded by default.",
                            deprecated))
    if n > curated_threshold:
        checks.append(Check("TOO_MANY_TOOLS", "warning",
                            f"{n} operations would flood the model context; curated mode is recommended.", n))

    poisoned = 0
    for o in ops:
        findings = scan(o.summary + " " + o.description) + [f for p in o.params for f in scan(p.description)]
        if any(f.severity == "high" for f in findings):
            poisoned += 1
            o.warnings.append("possible tool poisoning: " + ", ".join(sorted({f.rule for f in findings})))
    if poisoned:
        checks.append(Check("TOOL_POISONING_SUSPECTED", "error",
                            "Descriptions contain instruction-like or secret-seeking text; sanitised.", poisoned))

    score = 100.0
    for c in checks:
        weight = WEIGHTS[c.severity]
        if c.affected and n:
            weight *= min(1.0, 0.4 + c.affected / n)
        score -= weight
    stats = {
        "operations": n,
        "methods": dict(Counter(o.method for o in ops)),
        "tags": len({t for o in ops for t in o.tags}),
        "parameters": len(params),
        "with_request_body": sum(o.body_schema is not None for o in ops),
        "security_schemes": list(spec.security_schemes),
        "format": spec.source_format,
    }
    return DoctorReport(score=max(0, round(score)), checks=checks, stats=stats)
