"""Command line interface.

    mcpforge generate spec.yaml -o out/            generate an MCP server project
    mcpforge doctor spec.yaml                      spec health report
    mcpforge classify spec.yaml                    risk label per operation
    mcpforge scan spec.yaml                        tool-poisoning scan
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .doctor import diagnose
from .generator import GenerateOptions, generate
from .loader import load_spec
from .poisoning import scan
from .risk import RiskClassifier


def _cmd_generate(a: argparse.Namespace) -> int:
    opts = GenerateOptions(output=Path(a.output), mode=a.mode, max_tools=a.max_tools, policy=a.policy,
                           include_deprecated=a.include_deprecated, enhance=a.enhance,
                           bedrock_model_id=a.bedrock_model_id, embeddings=a.embeddings, server_name=a.name)
    res = generate(a.spec, opts)
    r = res.report
    print(f"Generated {res.output}/server.py for '{r['api']}'")
    print(f"  operations: {r['operations']}  direct tools: {r['direct_tools']} ({r['mode']} mode)")
    print(f"  risk: {r['risk_counts']}  classifier: {r['classifier']}")
    print(f"  spec health: {r['doctor']['score']}/100  drift: {r['drift'].get('status')}")
    if r["needs_review"]:
        print(f"  {len(r['needs_review'])} low-confidence labels listed in mcpforge_report.json")
    if a.publish_metrics:
        from .aws.cloudwatch import publish

        ok = publish({"Operations": r["operations"], "DirectTools": r["direct_tools"],
                      "SpecHealth": r["doctor"]["score"],
                      "Destructive": r["risk_counts"]["destructive"],
                      "LowConfidence": len(r["needs_review"]),
                      **{k: v for k, v in r["drift"].items() if isinstance(v, (int, float))}}, r["api"])
        print("  CloudWatch metrics:", "published" if ok else "skipped (no AWS credentials)")
    return 0


def _cmd_doctor(a: argparse.Namespace) -> int:
    report = diagnose(load_spec(a.spec))
    if a.json:
        print(json.dumps(report.to_dict(), indent=2))
        return 0
    print(f"Spec health: {report.score}/100  ({report.stats['operations']} operations, {report.stats['format']})")
    for c in report.checks:
        extra = f" [{c.affected}]" if c.affected else ""
        print(f"  {c.severity.upper():7} {c.code}{extra}: {c.message}")
    return 0 if report.score >= a.fail_under else 1


def _cmd_classify(a: argparse.Namespace) -> int:
    spec = load_spec(a.spec)
    clf = RiskClassifier()
    preds = clf.annotate(spec.operations)
    for op, p in zip(spec.operations, preds):
        flag = " (review)" if p.needs_review else ""
        print(f"{op.risk:12} {p.confidence:5.2f}  {op.method:6} {op.path}  [{op.operation_id}]{flag}")
    return 0


def _cmd_scan(a: argparse.Namespace) -> int:
    spec = load_spec(a.spec)
    hits = 0
    for op in spec.operations:
        for f in scan(op.summary + " " + op.description) + [f for p in op.params for f in scan(p.description)]:
            hits += 1
            print(f"{f.severity:6} {op.operation_id}: {f.rule}: {f.excerpt!r}")
    print(f"{hits} finding(s)")
    return 1 if hits else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mcpforge", description="OpenAPI to MCP server generator")
    ap.add_argument("--version", action="version", version=f"mcpforge {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate an MCP server project")
    g.add_argument("spec")
    g.add_argument("-o", "--output", required=True)
    g.add_argument("--mode", choices=["auto", "all", "curated"], default="auto")
    g.add_argument("--max-tools", type=int, default=25)
    g.add_argument("--policy", choices=["read_only", "read_write", "full"], default="read_write")
    g.add_argument("--include-deprecated", action="store_true")
    g.add_argument("--enhance", choices=["bedrock"], default=None, help="rewrite descriptions with Bedrock")
    g.add_argument("--bedrock-model-id", default="amazon.nova-lite-v1:0")
    g.add_argument("--embeddings", choices=["bedrock"], default=None, help="Titan embeddings for the router")
    g.add_argument("--name", default=None, help="server name")
    g.add_argument("--publish-metrics", action="store_true", help="send metrics to CloudWatch")
    g.set_defaults(func=_cmd_generate)

    d = sub.add_parser("doctor", help="spec health report")
    d.add_argument("spec")
    d.add_argument("--json", action="store_true")
    d.add_argument("--fail-under", type=int, default=0)
    d.set_defaults(func=_cmd_doctor)

    c = sub.add_parser("classify", help="risk label per operation")
    c.add_argument("spec")
    c.set_defaults(func=_cmd_classify)

    s = sub.add_parser("scan", help="tool-poisoning scan")
    s.add_argument("spec")
    s.set_defaults(func=_cmd_scan)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
