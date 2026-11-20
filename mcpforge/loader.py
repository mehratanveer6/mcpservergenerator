"""Load OpenAPI 3.x / Swagger 2.0 documents (JSON or YAML) into the IR.

Responsibilities:
  * read the file and detect the dialect
  * resolve local JSON references ("#/components/schemas/Pet") with cycle protection
  * work out a usable base URL (server variables, Swagger host + basePath)
  * flatten path-level and operation-level parameters
  * normalise request bodies (OpenAPI requestBody, Swagger body / formData params)
  * synthesise stable operation ids when a spec does not provide them
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import yaml

from .ir import ApiSpec, Operation, Param

HTTP_METHODS = ("get", "put", "post", "delete", "patch", "head", "options")
MAX_REF_DEPTH = 6
MAX_LEVEL = 10  # path item -> method -> requestBody -> content -> media -> schema -> properties -> field


class SpecError(ValueError):
    """Raised when a document cannot be interpreted as an API description."""


class _SpecLoader(yaml.SafeLoader):
    """SafeLoader that keeps timestamps as strings (specs are not data files)."""


_SpecLoader.yaml_implicit_resolvers = {
    k: [(tag, rx) for tag, rx in v if tag != "tag:yaml.org,2002:timestamp"]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def read_document(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    text = _CONTROL.sub(" ", text)
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        doc = yaml.load(text, Loader=_SpecLoader)  # noqa: S506 - SafeLoader subclass
    if not isinstance(doc, dict):
        raise SpecError(f"{path} is not a JSON/YAML object")
    return doc


def _lookup(doc: dict[str, Any], ref: str) -> Any:
    if not ref.startswith("#/"):
        raise KeyError(ref)  # remote refs are out of scope; caller degrades gracefully
    node: Any = doc
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        node = node[part]
    return node


def resolve_refs(node: Any, doc: dict[str, Any], seen: tuple[str, ...] = (), max_depth: int = MAX_REF_DEPTH,
                 level: int = 0, max_level: int = MAX_LEVEL) -> Any:
    """Return a copy of node with local $refs inlined, bounded in depth and size.

    Two limits keep large real-world specs (Stripe, GitHub) tractable:
      * max_depth: number of nested $ref hops; recursive schemas stop when a ref repeats
      * max_level: structural nesting; deeper sub-schemas collapse to a type stub,
        because a tool signature only needs the top-level fields of a request body
    """
    if level > max_level:
        if isinstance(node, dict):
            t = node.get("type")
            return {"type": t if isinstance(t, str) else "object"}
        return [] if isinstance(node, list) else copy.copy(node)
    if isinstance(node, dict):
        if "$ref" in node and isinstance(node["$ref"], str):
            ref = node["$ref"]
            if ref in seen or len(seen) >= max_depth:
                return {"type": "object", "description": f"nested reference to {ref.split('/')[-1]}"}
            try:
                target = _lookup(doc, ref)
            except (KeyError, TypeError, IndexError):
                return {"type": "object", "description": f"unresolved reference {ref}"}
            merged = resolve_refs(target, doc, seen + (ref,), max_depth, level, max_level)
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            if siblings and isinstance(merged, dict):
                merged = {**merged, **resolve_refs(siblings, doc, seen, max_depth, level, max_level)}
            return merged
        return {k: resolve_refs(v, doc, seen, max_depth, level + 1, max_level) for k, v in node.items()}
    if isinstance(node, list):
        return [resolve_refs(v, doc, seen, max_depth, level + 1, max_level) for v in node]
    return copy.copy(node)


def _detect_format(doc: dict[str, Any]) -> str:
    if str(doc.get("swagger", "")).startswith("2"):
        return "swagger-2.0"
    ver = str(doc.get("openapi", ""))
    if ver.startswith("3.1"):
        return "openapi-3.1"
    if ver.startswith("3"):
        return "openapi-3.0"
    raise SpecError("document has neither an 'openapi' nor a 'swagger' version field")


def _base_url(doc: dict[str, Any], fmt: str) -> tuple[str, str]:
    """Return (usable_base_url, raw_value_from_spec)."""
    if fmt == "swagger-2.0":
        host = doc.get("host", "")
        base = doc.get("basePath", "") or ""
        schemes = doc.get("schemes") or ["https"]
        scheme = "https" if "https" in schemes else schemes[0]
        raw = f"{host}{base}"
        return (f"{scheme}://{host}{base}".rstrip("/") if host else base.rstrip("/")), raw
    servers = doc.get("servers") or []
    if not servers:
        return "", ""
    server = servers[0]
    url = server.get("url", "")
    raw = url
    for name, var in (server.get("variables") or {}).items():
        url = url.replace("{" + name + "}", str(var.get("default", "")))
    return url.rstrip("/"), raw


_SLUG = re.compile(r"[^a-zA-Z0-9]+")


def synthesize_operation_id(method: str, path: str) -> str:
    parts = [p for p in path.strip("/").split("/") if p]
    words = []
    for p in parts:
        if p.startswith("{") and p.endswith("}"):
            words.append("by_" + p[1:-1])
        else:
            words.append(p)
    slug = _SLUG.sub("_", "_".join(words)).strip("_").lower()
    return f"{method.lower()}_{slug or 'root'}"


def _swagger2_param_schema(p: dict[str, Any]) -> dict[str, Any]:
    keys = ("type", "format", "items", "enum", "default", "minimum", "maximum", "pattern")
    return {k: p[k] for k in keys if k in p}


def _parse_params(raw_params: list[dict[str, Any]], fmt: str) -> tuple[list[Param], dict[str, Any] | None, bool, str]:
    params: list[Param] = []
    body: dict[str, Any] | None = None
    body_required = False
    content_type = "application/json"
    form_props: dict[str, Any] = {}
    form_required: list[str] = []
    for p in raw_params:
        if not isinstance(p, dict) or "name" not in p:
            continue
        loc = p.get("in", "query")
        if fmt == "swagger-2.0" and loc == "body":
            body = p.get("schema") or {"type": "object"}
            body_required = bool(p.get("required", False))
            continue
        if fmt == "swagger-2.0" and loc == "formData":
            form_props[p["name"]] = _swagger2_param_schema(p)
            if p.get("description"):
                form_props[p["name"]]["description"] = p["description"]
            if p.get("required"):
                form_required.append(p["name"])
            continue
        schema = p.get("schema") if fmt != "swagger-2.0" else _swagger2_param_schema(p)
        params.append(
            Param(
                name=p["name"],
                location=loc,
                required=bool(p.get("required", loc == "path")),
                schema=schema or {"type": "string"},
                description=(p.get("description") or "").strip(),
            )
        )
    if form_props and body is None:
        body = {"type": "object", "properties": form_props, "required": form_required}
        body_required = bool(form_required)
        content_type = "application/x-www-form-urlencoded"
    return params, body, body_required, content_type


def _parse_request_body(rb: dict[str, Any]) -> tuple[dict[str, Any] | None, bool, str]:
    content = rb.get("content") or {}
    if not content:
        return None, False, "application/json"
    preferred = [c for c in content if "json" in c] or list(content)
    ctype = preferred[0]
    schema = (content.get(ctype) or {}).get("schema") or {"type": "object"}
    return schema, bool(rb.get("required", False)), ctype


def load_spec(path: str | Path, max_ref_depth: int = MAX_REF_DEPTH) -> ApiSpec:
    return parse_spec(read_document(path), max_ref_depth=max_ref_depth)


def parse_spec(doc: dict[str, Any], max_ref_depth: int = MAX_REF_DEPTH) -> ApiSpec:
    fmt = _detect_format(doc)
    info = doc.get("info") or {}
    base_url, raw_url = _base_url(doc, fmt)
    if fmt == "swagger-2.0":
        schemes = resolve_refs(doc.get("securityDefinitions") or {}, doc)
    else:
        schemes = resolve_refs((doc.get("components") or {}).get("securitySchemes") or {}, doc)
    global_security = [name for req in (doc.get("security") or []) if isinstance(req, dict) for name in req]

    spec = ApiSpec(
        title=str(info.get("title") or "api"),
        version=str(info.get("version") or "0"),
        base_url=base_url,
        raw_server_url=raw_url,
        source_format=fmt,
        security_schemes=schemes if isinstance(schemes, dict) else {},
        description=str(info.get("description") or "")[:2000],
    )

    used_ids: set[str] = set()
    for path, item in (doc.get("paths") or {}).items():
        if not isinstance(item, dict):
            continue
        item = resolve_refs(item, doc, max_depth=max_ref_depth)
        shared = item.get("parameters") or []
        for method in HTTP_METHODS:
            raw = item.get(method)
            if not isinstance(raw, dict):
                continue
            # operation-level parameters override path-level ones with the same (name, in)
            merged: dict[tuple[str, str], dict[str, Any]] = {}
            for p in list(shared) + list(raw.get("parameters") or []):
                if isinstance(p, dict) and "name" in p:
                    merged[(p["name"], p.get("in", "query"))] = p
            params, body, body_req, ctype = _parse_params(list(merged.values()), fmt)
            if fmt != "swagger-2.0" and isinstance(raw.get("requestBody"), dict):
                body, body_req, ctype = _parse_request_body(raw["requestBody"])

            op_id = str(raw.get("operationId") or "").strip()
            synthesized = not op_id
            if synthesized:
                op_id = synthesize_operation_id(method, path)
            unique = op_id
            n = 2
            while unique in used_ids:
                unique = f"{op_id}_{n}"
                n += 1
            used_ids.add(unique)

            sec = raw.get("security")
            op_security = [name for req in sec for name in req] if isinstance(sec, list) else global_security
            spec.operations.append(
                Operation(
                    operation_id=unique,
                    method=method.upper(),
                    path=path,
                    summary=str(raw.get("summary") or "").strip(),
                    description=str(raw.get("description") or "").strip(),
                    tags=[str(t) for t in (raw.get("tags") or [])],
                    params=params,
                    body_schema=body,
                    body_required=body_req,
                    body_content_type=ctype,
                    security=op_security,
                    deprecated=bool(raw.get("deprecated", False)),
                    synthesized_id=synthesized,
                )
            )
    if not spec.operations:
        raise SpecError("no operations found under 'paths'")
    return spec
