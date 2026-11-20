"""End-to-end generation pipeline.

    spec file -> load -> doctor -> risk classification -> names and descriptions
              -> (optional Bedrock enhancement / embeddings) -> curation
              -> drift check -> catalog.json + server.py + project files + report
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import __version__
from .curate import select
from .describe import assign_names_and_descriptions, enhance_with_bedrock, python_identifier, snake
from .doctor import diagnose
from .drift import check_drift
from .features import frame_from_operations
from .ir import ApiSpec, Operation
from .loader import load_spec
from .poisoning import sanitize, scan
from .risk import RiskClassifier, annotation_hints

TEMPLATES = Path(__file__).parent / "templates"
RUNTIME = Path(__file__).parent / "runtime"
POLICIES = {"read_only": {"read"}, "read_write": {"read", "write"}, "full": {"read", "write", "destructive"}}
PY_TYPES = {"integer": "int", "number": "float", "boolean": "bool", "string": "str",
            "array": "list[Any]", "object": "dict[str, Any]"}
MAX_FLAT_FIELDS = 12


@dataclass
class GenerateOptions:
    output: Path
    mode: str = "auto"  # auto | all | curated
    max_tools: int = 25
    policy: str = "read_write"
    include_deprecated: bool = False
    enhance: str | None = None  # None | "bedrock"
    bedrock_model_id: str = "amazon.nova-lite-v1:0"
    embeddings: str | None = None  # None | "bedrock"
    model_path: str | None = None
    server_name: str | None = None


@dataclass
class GenerateResult:
    output: Path
    spec: ApiSpec
    direct_tools: list[str]
    report: dict[str, Any] = field(default_factory=dict)


def _json_type(schema: dict[str, Any]) -> str:
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    if not t:
        t = "object" if "properties" in schema else "array" if "items" in schema else "string"
    return t if t in PY_TYPES else "string"


def _describe_schema(schema: dict[str, Any], base: str) -> str:
    text = base or schema.get("description", "")
    text = "" if any(f.severity in ("high", "medium") for f in scan(text)) else sanitize(text, 240)
    if text and not text.endswith((".", "!", "?")):
        text += "."
    if schema.get("enum"):
        text = (text + " " if text else "") + "One of: " + ", ".join(map(str, schema["enum"][:15])) + "."
    if "default" in schema and not isinstance(schema["default"], (dict, list)):
        text = (text + " " if text else "") + f"Default: {schema['default']}."
    return text


def build_arguments(op: Operation) -> tuple[list[dict], dict | None]:
    """Return (argument list for the Python signature, body description for the catalog)."""
    args: list[dict] = []
    used: set[str] = set()

    def unique(name: str, prefix: str) -> str:
        ident = python_identifier(name)
        if ident in used:
            ident = python_identifier(f"{prefix}_{name}")
        while ident in used:
            ident += "_"
        used.add(ident)
        return ident

    for p in op.params:
        if p.location == "header" and p.name.lower() in ("authorization", "accept", "content-type"):
            continue
        args.append({
            "arg": unique(p.name, p.location), "name": p.name, "in": p.location, "required": p.required,
            "type": p.json_type, "description": _describe_schema(p.schema, p.description),
        })

    body = None
    schema = op.body_schema
    if schema is not None:
        props = schema.get("properties") if isinstance(schema, dict) else None
        if _json_type(schema) == "object" and props and len(props) <= MAX_FLAT_FIELDS:
            required = set(schema.get("required") or []) if op.body_required else set()
            fields = []
            for name, sub in props.items():
                sub = sub if isinstance(sub, dict) else {}
                if sub.get("readOnly"):
                    continue
                fields.append({
                    "arg": unique(name, "body"), "name": name, "required": name in required,
                    "type": _json_type(sub), "description": _describe_schema(sub, sub.get("description", "")),
                })
            body = {"mode": "fields", "fields": fields, "required": op.body_required,
                    "content_type": op.body_content_type}
            args += [{**f, "in": "body"} for f in fields]
        else:
            used.add("body")
            body = {"mode": "whole", "fields": [], "required": op.body_required,
                    "content_type": op.body_content_type}
            desc = sanitize(schema.get("description", "") if isinstance(schema, dict) else "", 200)
            args.append({"arg": "body", "name": "body", "in": "body", "required": op.body_required,
                         "type": _json_type(schema) if isinstance(schema, dict) else "object",
                         "description": desc or "Request body as JSON."})
    return args, body


def _signature(args: list[dict]) -> str:
    parts_required, parts_optional = [], []
    for a in args:
        py = PY_TYPES[a["type"]] if a["type"] in PY_TYPES else "str"
        desc = repr(a["description"] or a["name"])
        if a["required"]:
            parts_required.append(f"{a['arg']}: Annotated[{py}, Field(description={desc})]")
        else:
            parts_optional.append(f"{a['arg']}: Annotated[{py} | None, Field(description={desc})] = None")
    parts = parts_required + parts_optional
    return ("\n    " + ",\n    ".join(parts) + ",\n") if parts else ""


def _auth_entries(spec: ApiSpec) -> list[dict]:
    out = []
    for name, s in spec.security_schemes.items():
        if not isinstance(s, dict):
            continue
        kind = s.get("type")
        if kind == "apiKey":
            out.append({"scheme": name, "type": "apiKey", "in": s.get("in", "header"), "name": s.get("name", "X-API-Key")})
        elif kind == "http" and str(s.get("scheme", "")).lower() == "basic" or kind == "basic":
            out.append({"scheme": name, "type": "basic"})
        else:  # http bearer, oauth2, openIdConnect: a bearer token is the common denominator
            out.append({"scheme": name, "type": "bearer"})
    return out


def _env() -> Environment:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), undefined=StrictUndefined,
                      keep_trailing_newline=True, trim_blocks=False, lstrip_blocks=False)
    env.filters["pyrepr"] = repr
    return env


def generate(spec_path: str | Path, opts: GenerateOptions) -> GenerateResult:
    t0 = time.time()
    spec = load_spec(spec_path)
    if not opts.include_deprecated:
        spec.operations = [o for o in spec.operations if not o.deprecated] or spec.operations

    doctor = diagnose(spec, curated_threshold=opts.max_tools)
    classifier = RiskClassifier(opts.model_path)
    preds = classifier.annotate(spec.operations)
    assign_names_and_descriptions(spec)

    enhanced = 0
    if opts.enhance == "bedrock":
        enhanced = enhance_with_bedrock(spec, opts.bedrock_model_id)

    allowed = POLICIES[opts.policy]
    mode = opts.mode
    if mode == "auto":
        mode = "curated" if len(spec.operations) > opts.max_tools else "all"
    if mode == "all":
        direct_ops = [o for o in spec.operations]
    else:
        direct_ops = select(spec.operations, opts.max_tools, allowed)

    catalog_ops, direct = [], []
    direct_ids = {o.operation_id for o in direct_ops}
    for op in spec.operations:
        args, body = build_arguments(op)
        entry = {
            "operation_id": op.operation_id, "tool_name": op.tool_name, "method": op.method, "path": op.path,
            "summary": sanitize(op.summary, 200), "description": op.tool_description, "tags": op.tags,
            "risk": op.risk, "risk_confidence": op.risk_confidence,
            "params": [a for a in args if a["in"] != "body"], "body": body, "direct": op.operation_id in direct_ids,
        }
        catalog_ops.append(entry)
        if op.operation_id in direct_ids:
            hints = annotation_hints(op)
            hints["title"] = sanitize(op.summary, 80) or op.tool_name.replace("_", " ")
            direct.append({
                "tool_name": op.tool_name, "operation_id": op.operation_id, "description": op.tool_description,
                "risk": op.risk, "hints": hints, "args": args, "signature": _signature(args),
            })

    if opts.embeddings == "bedrock":
        from .aws.bedrock import embed_many
        from .runtime.router import document_for

        vectors = embed_many([document_for(e) for e in catalog_ops])
        for e, v in zip(catalog_ops, vectors):
            e["vector"] = v

    drift = check_drift(frame_from_operations(spec.operations), classifier.model)
    out = Path(opts.output)
    out.mkdir(parents=True, exist_ok=True)
    server_name = opts.server_name or snake(spec.title)[:40] or "api"
    title = re.sub(r"[\"\\]", "", spec.title)
    catalog = {"title": spec.title, "version": spec.version, "base_url": spec.base_url,
               "auth": _auth_entries(spec), "operations": catalog_ops}
    (out / "catalog.json").write_text(json.dumps(catalog, indent=1), encoding="utf-8")
    shutil.copy(RUNTIME / "router.py", out / "router.py")

    risk_counts = {r: sum(o.risk == r for o in spec.operations) for r in ("read", "write", "destructive")}
    instructions = (
        f"Tools for the {title} API ({len(spec.operations)} operations). "
        + ("Use search_operations to find operations that are not listed as tools, then call_operation. "
           if len(direct) < len(spec.operations) else "")
        + "Tools marked destructive delete or reverse data: confirm with the user before calling them."
    )
    ctx = {
        "title": title, "forge_version": __version__, "source_name": Path(spec_path).name,
        "n_ops": len(spec.operations), "direct": direct, "base_url": spec.base_url,
        "server_name": server_name, "api_version": spec.version, "instructions": instructions,
        "policy": opts.policy, "risk_counts": risk_counts, "auth": catalog["auth"], "mode": mode,
    }
    env = _env()
    for template, target in [("server.py.j2", "server.py"), ("README.md.j2", "README.md"),
                             ("pyproject.toml.j2", "pyproject.toml"), ("Dockerfile.j2", "Dockerfile"),
                             ("env.example.j2", ".env.example")]:
        (out / target).write_text(env.get_template(template).render(**ctx), encoding="utf-8")

    report = {
        "api": spec.title, "source": str(spec_path), "format": spec.source_format,
        "operations": len(spec.operations), "mode": mode, "direct_tools": len(direct), "policy": opts.policy,
        "risk_counts": risk_counts, "classifier": "model" if classifier.available else "rules",
        "needs_review": [o.operation_id for o, p in zip(spec.operations, preds) if p.needs_review],
        "doctor": doctor.to_dict(), "drift": drift, "bedrock_descriptions_accepted": enhanced,
        "warnings": {o.operation_id: o.warnings for o in spec.operations if o.warnings},
        "seconds": round(time.time() - t0, 2),
    }
    (out / "mcpforge_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return GenerateResult(out, spec, [d["tool_name"] for d in direct], report)
