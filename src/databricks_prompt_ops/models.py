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
    tags: list[str] = field(default_factory=list)


@dataclass
class PromptRegistration:
    prompt_id: str
    user_id: str
    session_id: str
    pipeline_type: str
    prompt_text: str
    registered_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptValidationReport:
    is_valid: bool
    issues: list[str] = field(default_factory=list)
    normalized_prompt: str = ""


@dataclass
class PromptEvaluationReport:
    safety_score: float
    reliability_score: float
    fairness_score: float
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

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
