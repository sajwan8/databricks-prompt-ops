from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import tomllib


class PipelineType(str, Enum):
    RAG = "rag"
    AGENTIC_AI = "agentic_ai"
    GENERATIVE_AI = "generative_ai"


class StorageBackend(str, Enum):
    JSON = "json"
    DELTA = "delta"


@dataclass
class PipelineSettings:
    name: str
    type: PipelineType
    environment: str


@dataclass
class ModelSettings:
    llm_model: str
    fallback_model: str


@dataclass
class PromptSettings:
    registry_path: str
    request_store_path: str
    evaluation_store_path: str
    clarification_prompt_name: str
    llm_prompt_name: str
    rag_prompt_name: str
    agentic_prompt_name: str


@dataclass
class DatabricksSettings:
    workspace_url: str
    serving_endpoint: str


@dataclass
class StorageSettings:
    backend: StorageBackend = StorageBackend.JSON
    catalog: str | None = None
    schema: str | None = None
    registry_table: str = "prompt_registry"
    request_table: str = "prompt_requests"
    evaluation_table: str = "prompt_evaluations"
    prefer_databricks_connect: bool = True


@dataclass
class ValidationSettings:
    min_prompt_length: int
    require_question_for_rag: bool
    require_goal_for_agentic: bool
    block_unsafe_requests: bool
    max_prompt_length: int = 4000
    block_sensitive_data: bool = True
    normalize_whitespace: bool = True


@dataclass
class EvaluationSettings:
    run_evaluation: bool
    min_safety_score: float
    min_reliability_score: float
    min_fairness_score: float
    min_overall_score: float = 0.70


@dataclass
class PromptOpsConfig:
    pipeline: PipelineSettings
    models: ModelSettings
    prompts: PromptSettings
    storage: StorageSettings
    databricks: DatabricksSettings
    validation: ValidationSettings
    evaluation: EvaluationSettings


def load_config(config_path: str | Path) -> PromptOpsConfig:
    with Path(config_path).open("rb") as file:
        raw = tomllib.load(file)

    storage_raw = raw.get("storage", {})
    validation_raw = raw["validation"]
    evaluation_raw = raw["evaluation"]

    return PromptOpsConfig(
        pipeline=PipelineSettings(
            name=raw["pipeline"]["name"],
            type=PipelineType(raw["pipeline"]["type"]),
            environment=raw["pipeline"]["environment"],
        ),
        models=ModelSettings(**raw["models"]),
        prompts=PromptSettings(**raw["prompts"]),
        storage=StorageSettings(
            backend=StorageBackend(storage_raw.get("backend", StorageBackend.JSON.value)),
            catalog=storage_raw.get("catalog"),
            schema=storage_raw.get("schema"),
            registry_table=storage_raw.get("registry_table", "prompt_registry"),
            request_table=storage_raw.get("request_table", "prompt_requests"),
            evaluation_table=storage_raw.get("evaluation_table", "prompt_evaluations"),
            prefer_databricks_connect=storage_raw.get("prefer_databricks_connect", True),
        ),
        databricks=DatabricksSettings(**raw["databricks"]),
        validation=ValidationSettings(
            min_prompt_length=validation_raw["min_prompt_length"],
            max_prompt_length=validation_raw.get("max_prompt_length", 4000),
            require_question_for_rag=validation_raw["require_question_for_rag"],
            require_goal_for_agentic=validation_raw["require_goal_for_agentic"],
            block_unsafe_requests=validation_raw["block_unsafe_requests"],
            block_sensitive_data=validation_raw.get("block_sensitive_data", True),
            normalize_whitespace=validation_raw.get("normalize_whitespace", True),
        ),
        evaluation=EvaluationSettings(
            run_evaluation=evaluation_raw["run_evaluation"],
            min_safety_score=evaluation_raw["min_safety_score"],
            min_reliability_score=evaluation_raw["min_reliability_score"],
            min_fairness_score=evaluation_raw["min_fairness_score"],
            min_overall_score=evaluation_raw.get("min_overall_score", 0.70),
        ),
    )
