"""Train, tune and deploy the risk classifier on Amazon SageMaker AI.

    python aws/sagemaker_train_deploy.py --role arn:aws:iam::<account>:role/MCPForgeSageMakerRole

1. uploads ml/data to the session's default S3 bucket
2. runs ml/train.py in the managed scikit-learn container (script mode) on Spot capacity
3. Automatic Model Tuning (Bayesian) over C, n-gram range and vocabulary size,
   optimising the gold-set macro F1 printed by train.py
4. deploys the best model to a serverless inference endpoint (pay per request,
   scales to zero) with ml/inference.py as the serving script
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sagemaker
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.tuner import CategoricalParameter, ContinuousParameter, HyperparameterTuner, IntegerParameter

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = "1.2-1"
METRIC = "validation:macro_f1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default=None, help="SageMaker execution role ARN")
    ap.add_argument("--max-jobs", type=int, default=12)
    ap.add_argument("--endpoint-name", default="mcpforge-risk-classifier")
    ap.add_argument("--skip-deploy", action="store_true")
    a = ap.parse_args()

    session = sagemaker.Session()
    role = a.role or sagemaker.get_execution_role()
    bucket = session.default_bucket()
    data_uri = session.upload_data(str(ROOT / "ml" / "data"), bucket=bucket, key_prefix="mcpforge/data")
    print("training data:", data_uri)

    estimator = SKLearn(
        entry_point="train.py",
        source_dir=str(ROOT / "ml"),
        dependencies=[str(ROOT / "mcpforge")],
        framework_version=FRAMEWORK,
        py_version="py3",
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        use_spot_instances=True,
        max_run=1800,
        max_wait=3600,
        output_path=f"s3://{bucket}/mcpforge/models",
        hyperparameters={"C": 8.0, "ngram-max": 1, "max-features": 5000},
        metric_definitions=[{"Name": METRIC, "Regex": r"validation:macro_f1=([0-9\.]+)"}],
        tags=[{"Key": "project", "Value": "mcpforge"}],
    )
    tuner = HyperparameterTuner(
        estimator,
        objective_metric_name=METRIC,
        objective_type="Maximize",
        hyperparameter_ranges={
            "C": ContinuousParameter(0.25, 16.0, scaling_type="Logarithmic"),
            "ngram-max": IntegerParameter(1, 2),
            "max-features": CategoricalParameter([5000, 20000]),
        },
        metric_definitions=[{"Name": METRIC, "Regex": r"validation:macro_f1=([0-9\.]+)"}],
        strategy="Bayesian",
        max_jobs=a.max_jobs,
        max_parallel_jobs=3,
        early_stopping_type="Auto",
    )
    tuner.fit({"train": data_uri, "validation": data_uri})
    best = tuner.best_training_job()
    print("best training job:", best)
    if a.skip_deploy:
        return

    model_data = sagemaker.estimator.Estimator.attach(best, sagemaker_session=session).model_data
    model = SKLearnModel(
        model_data=model_data,
        role=role,
        entry_point="inference.py",
        source_dir=str(ROOT / "ml"),
        dependencies=[str(ROOT / "mcpforge")],
        framework_version=FRAMEWORK,
        py_version="py3",
    )
    predictor = model.deploy(
        endpoint_name=a.endpoint_name,
        serverless_inference_config=ServerlessInferenceConfig(memory_size_in_mb=2048, max_concurrency=5),
    )
    print("endpoint:", predictor.endpoint_name)


if __name__ == "__main__":
    main()
