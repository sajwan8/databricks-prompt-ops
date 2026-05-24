from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any


@dataclass
class PromptTemplate:
    name: str
    version: str
    description: str
    template: str
    source_path: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class PromptRegistration:
    prompt_id: str
    user_id: str
    session_id: str
    pipeline_type: str
    raw_user_input: str
    modified_prompt: str
    registered_at: str
    validation_approach: str = "rule_based_validation"
    inference_model_name: str | None = None
    prompt_template_name: str | None = None
    prompt_template_version: str | None = None
    prompt_template_source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationStageResult:
    stage_name: str
    is_valid: bool
    issues: list[str] = field(default_factory=list)
    details: str = ""


@dataclass
class PromptValidationReport:
    is_valid: bool
    issues: list[str] = field(default_factory=list)
    normalized_prompt: str = ""
    validation_approach: str = "rule_based_validation"
    stage_results: list[ValidationStageResult] = field(default_factory=list)


@dataclass
class PromptEvaluationReport:
    safety_score: float
    reliability_score: float
    fairness_score: float
    toxicity_score: float
    correctness_score: float
    completeness_score: float
    consistency_score: float
    relevance_score: float
    overall_score: float
    notes: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class PromptOpsResponse:
    status: str
    pipeline_type: str
    prompt_id: str
    assistant_message: str
    registered_prompt: PromptRegistration
    validation_report: PromptValidationReport
    evaluation_report: PromptEvaluationReport | None = None
    routed_downstream: bool = False
    llm_response: str | None = None
    inference_model_name: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class ModelCompletionResult:
    content: str
    model_name: str
