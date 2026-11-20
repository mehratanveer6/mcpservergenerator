"""Intermediate representation (IR) for API operations.

Every input format (OpenAPI 3.x, Swagger 2.0, JSON or YAML) is normalised into
these dataclasses before any analysis or code generation happens. Keeping one IR
means the doctor, the risk classifier, the router and the code generator never
need to know which spec dialect the user supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Param:
    name: str
    location: str  # path | query | header | cookie | body
    required: bool = False
    schema: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    @property
    def json_type(self) -> str:
        t = self.schema.get("type")
        if isinstance(t, list):  # OpenAPI 3.1 allows ["string", "null"]
            t = next((x for x in t if x != "null"), "string")
        if not t:
            if "properties" in self.schema:
                return "object"
            if "items" in self.schema:
                return "array"
            return "string"
        return t


@dataclass
class Operation:
    operation_id: str
    method: str
    path: str
    summary: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    params: list[Param] = field(default_factory=list)
    body_schema: dict[str, Any] | None = None
    body_required: bool = False
    body_content_type: str = "application/json"
    security: list[str] = field(default_factory=list)
    deprecated: bool = False
    synthesized_id: bool = False

    # filled in by the analysis stages
    risk: str = "unknown"  # read | write | destructive
    risk_confidence: float = 0.0
    tool_name: str = ""
    tool_description: str = ""
    warnings: list[str] = field(default_factory=list)

    def text(self) -> str:
        """Flat text used by the ML model and the router."""
        names = " ".join(p.name for p in self.params)
        return " ".join(
            x
            for x in [
                self.operation_id,
                self.summary,
                self.description[:500],
                self.path,
                " ".join(self.tags),
                names,
            ]
            if x
        )


@dataclass
class ApiSpec:
    title: str
    version: str
    base_url: str
    source_format: str  # openapi-3.0 | openapi-3.1 | swagger-2.0
    operations: list[Operation] = field(default_factory=list)
    security_schemes: dict[str, dict[str, Any]] = field(default_factory=dict)
    description: str = ""
    raw_server_url: str = ""

    def by_id(self) -> dict[str, Operation]:
        return {op.operation_id: op for op in self.operations}
