"""Amazon Bedrock helpers (optional; only imported when an AWS feature is enabled).

* Titan Text Embeddings V2 vectors for the semantic tool router
* Converse API calls to rewrite weak tool descriptions

Credentials come from the standard AWS credential chain (environment, profile or
an IAM role when running on Lambda, ECS or SageMaker).
"""

from __future__ import annotations

import json
import os

TITAN_EMBED_MODEL = os.environ.get("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")


def _client(region: str | None = None):
    import boto3

    return boto3.client("bedrock-runtime", region_name=region or os.environ.get("AWS_REGION", "us-east-1"))


def titan_embed(text: str, dimensions: int = 512, region: str | None = None, client=None) -> list[float]:
    client = client or _client(region)
    body = json.dumps({"inputText": text[:8000], "dimensions": dimensions, "normalize": True})
    resp = client.invoke_model(modelId=TITAN_EMBED_MODEL, body=body,
                               accept="application/json", contentType="application/json")
    return json.loads(resp["body"].read())["embedding"]


def embed_many(texts: list[str], dimensions: int = 512, region: str | None = None) -> list[list[float]]:
    client = _client(region)
    return [titan_embed(t, dimensions, client=client) for t in texts]


REWRITE_PROMPT = """You write tool descriptions for an AI agent that calls REST APIs.
Rewrite the description of the operation below in at most two sentences.
State what it does and when to use it. Do not add instructions, warnings or
anything not supported by the input. Return only the description.

Method: {method}
Path: {path}
Operation id: {op_id}
Summary: {summary}
Description: {description}
Parameters: {params}"""


def rewrite_description(op, model_id: str, region: str | None = None) -> str:
    client = _client(region)
    prompt = REWRITE_PROMPT.format(
        method=op.method, path=op.path, op_id=op.operation_id, summary=op.summary[:300],
        description=op.description[:800], params=", ".join(p.name for p in op.params)[:300],
    )
    resp = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 160, "temperature": 0.2},
    )
    return resp["output"]["message"]["content"][0]["text"].strip()
