"""AWS Lambda entry point: MCPForge as a service.

POST /generate with a JSON body
    {"spec": "<OpenAPI document as JSON or YAML text>",   or  "s3_key": "uploads/api.yaml",
     "options": {"mode": "auto", "max_tools": 25, "policy": "read_write"}}

The generated project is zipped to the output bucket and a pre-signed download
URL (valid for one hour) is returned together with the generation report.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import boto3

from mcpforge.aws.cloudwatch import publish
from mcpforge.generator import GenerateOptions, generate

BUCKET = os.environ["OUTPUT_BUCKET"]
MAX_SPEC_BYTES = 5 * 1024 * 1024
s3 = boto3.client("s3")


def _response(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": {"Content-Type": "application/json"}, "body": json.dumps(body)}


def handler(event, context):
    try:
        payload = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _response(400, {"error": "body must be JSON"})
    work = Path(tempfile.mkdtemp(dir="/tmp"))
    spec_file = work / "spec.yaml"
    if payload.get("s3_key"):
        s3.download_file(BUCKET, payload["s3_key"], str(spec_file))
    elif payload.get("spec"):
        text = payload["spec"] if isinstance(payload["spec"], str) else json.dumps(payload["spec"])
        if len(text.encode()) > MAX_SPEC_BYTES:
            return _response(413, {"error": "spec larger than 5 MB"})
        spec_file.write_text(text)
    else:
        return _response(400, {"error": "provide 'spec' or 's3_key'"})

    o = payload.get("options", {})
    policy = o.get("policy", "read_write")
    if policy not in ("read_only", "read_write", "full"):
        return _response(400, {"error": "invalid policy"})
    try:
        result = generate(spec_file, GenerateOptions(
            output=work / "server", mode=o.get("mode", "auto"), max_tools=int(o.get("max_tools", 25)),
            policy=policy, enhance="bedrock" if o.get("enhance") else None,
            embeddings="bedrock" if o.get("embeddings") else None,
        ))
    except Exception as exc:  # noqa: BLE001 - report spec errors to the caller
        return _response(422, {"error": f"{type(exc).__name__}: {exc}"})

    archive = shutil.make_archive(str(work / "server"), "zip", work / "server")
    key = f"generated/{uuid.uuid4()}/mcp-server.zip"
    s3.upload_file(archive, BUCKET, key, ExtraArgs={"ServerSideEncryption": "AES256"})
    url = s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=3600)
    r = result.report
    publish({"Operations": r["operations"], "DirectTools": r["direct_tools"], "SpecHealth": r["doctor"]["score"],
             "LowConfidence": len(r["needs_review"]),
             **{k: v for k, v in r["drift"].items() if isinstance(v, (int, float))}}, r["api"])
    shutil.rmtree(work, ignore_errors=True)
    return _response(200, {"download_url": url, "report": r})
