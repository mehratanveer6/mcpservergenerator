"""SageMaker Pipelines definition: train -> evaluate -> quality gate -> register.

    python aws/sagemaker_pipeline.py --role <execution-role-arn> [--run]

The model is registered in the SageMaker Model Registry with status
PendingManualApproval only when the gold-set macro F1 reaches the threshold, so
a regression can never be promoted automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = "1.2-1"


def build(role: str) -> Pipeline:
    session = PipelineSession()
    bucket = session.default_bucket()
    data_uri = ParameterString("TrainingData", default_value=f"s3://{bucket}/mcpforge/data")
    min_f1 = ParameterFloat("MinMacroF1", default_value=0.90)

    estimator = SKLearn(
        entry_point="train.py", source_dir=str(ROOT / "ml"), dependencies=[str(ROOT / "mcpforge")],
        framework_version=FRAMEWORK, py_version="py3", role=role, instance_type="ml.m5.large",
        instance_count=1, sagemaker_session=session, use_spot_instances=True, max_run=1800, max_wait=3600,
        hyperparameters={"C": 8.0, "ngram-max": 1, "max-features": 5000},
    )
    train = TrainingStep(name="TrainRiskClassifier",
                         step_args=estimator.fit({"train": data_uri, "validation": data_uri}))

    processor = SKLearnProcessor(framework_version=FRAMEWORK, role=role, instance_type="ml.t3.medium",
                                 instance_count=1, sagemaker_session=session)
    evaluation = PropertyFile(name="Evaluation", output_name="evaluation", path="evaluation.json")
    evaluate = ProcessingStep(
        name="EvaluateOnGoldSet",
        step_args=processor.run(
            code=str(ROOT / "ml" / "evaluate.py"),
            inputs=[
                ProcessingInput(source=train.properties.ModelArtifacts.S3ModelArtifacts,
                                destination="/opt/ml/processing/model"),
                ProcessingInput(source=data_uri, destination="/opt/ml/processing/data"),
                ProcessingInput(source=str(ROOT / "mcpforge"), destination="/opt/ml/processing/input/code/mcpforge"),
            ],
            outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        ),
        property_files=[evaluation],
    )

    model = SKLearnModel(
        model_data=train.properties.ModelArtifacts.S3ModelArtifacts, role=role, entry_point="inference.py",
        source_dir=str(ROOT / "ml"), dependencies=[str(ROOT / "mcpforge")], framework_version=FRAMEWORK,
        py_version="py3", sagemaker_session=session,
    )
    register = ModelStep(
        name="RegisterRiskClassifier",
        step_args=model.register(
            content_types=["application/json"], response_types=["application/json"],
            inference_instances=["ml.t2.medium", "ml.m5.large"], transform_instances=["ml.m5.large"],
            model_package_group_name="mcpforge-risk-classifier", approval_status="PendingManualApproval",
        ),
    )
    gate = ConditionStep(
        name="QualityGate",
        conditions=[ConditionGreaterThanOrEqualTo(
            left=JsonGet(step_name=evaluate.name, property_file=evaluation, json_path="metrics.macro_f1.value"),
            right=min_f1,
        )],
        if_steps=[register],
        else_steps=[FailStep(name="RejectModel", error_message="macro F1 below threshold")],
    )
    return Pipeline(name="mcpforge-risk-pipeline", parameters=[data_uri, min_f1],
                    steps=[train, evaluate, gate], sagemaker_session=session)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True)
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    pipeline = build(a.role)
    pipeline.upsert(role_arn=a.role)
    print("pipeline upserted:", pipeline.name)
    if a.run:
        execution = pipeline.start()
        print("execution:", execution.arn)


if __name__ == "__main__":
    main()
