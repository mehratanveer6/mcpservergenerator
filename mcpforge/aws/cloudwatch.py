"""Publish generator and drift metrics to Amazon CloudWatch (optional)."""

from __future__ import annotations

import os

NAMESPACE = os.environ.get("MCPFORGE_METRIC_NAMESPACE", "MCPForge")


def publish(metrics: dict[str, float], api_name: str, region: str | None = None) -> bool:
    """Send numeric metrics with an ApiName dimension. Returns False if AWS is unavailable."""
    try:
        import boto3

        cw = boto3.client("cloudwatch", region_name=region or os.environ.get("AWS_REGION", "us-east-1"))
        data = [
            {"MetricName": k, "Value": float(v), "Unit": "None",
             "Dimensions": [{"Name": "ApiName", "Value": api_name[:250]}]}
            for k, v in metrics.items() if isinstance(v, (int, float))
        ]
        for i in range(0, len(data), 20):
            cw.put_metric_data(Namespace=NAMESPACE, MetricData=data[i:i + 20])
        return True
    except Exception:  # noqa: BLE001 - metrics must never break generation
        return False
