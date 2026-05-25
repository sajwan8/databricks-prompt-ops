from __future__ import annotations

import json
import re

from src.databricks_prompt_ops.config import PipelineType, ValidationSettings
from src.databricks_prompt_ops.models import PromptValidationReport, ValidationStageResult


UNSAFE_PATTERNS = [
    "bypass compliance",
    "avoid detection",
    "disable guardrails",
    "hide audit",
    "spoof identity",
]

SENSITIVE_DATA_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "Please remove Social Security numbers or equivalent national identifiers."),
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "Please remove payment card numbers from the prompt."),
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE), "Please avoid sending direct email addresses unless they are required and approved."),
]

GENERATION_HINTS = {"write", "draft", "summarize", "explain", "respond", "create", "prepare", "review", "help"}
RAG_HINTS = {"what", "how", "when", "where", "why", "which", "find", "lookup", "search", "retrieve"}

AGENTIC_HINTS = {"plan", "analyze", "investigate", "workflow", "steps", "goal", "orchestrate", "coordinate", "execute", "automate"}
SEMANTIC_CLARITY_HINTS = {"audience", "format", "summary", "timeline", "customer", "bullet", "steps", "details", "context"}

SEMANTIC_VALIDATION_PROMPT = """You are a prompt validation assistant.
Review the business prompt below for semantic quality only.
Return strict JSON with this schema:
{"is_valid": true|false, "issues": ["..."], "details": "short explanation"}

Semantic checks:
- Is the request coherent and internally understandable?
- Is it actionable enough for downstream execution?
- Does it appear meaningfully aligned to a business task?
- Is it missing essential context that makes the request ambiguous?

Business prompt:
{prompt_text}
"""



class PromptValidator:
    """Validates prompts using structural, security, domain, and semantic stages."""

    def __init__(self, settings: ValidationSettings, semantic_client=None) -> None:
        self.settings = settings
        self.semantic_client = semantic_client

    def _normalize_prompt(self, prompt_text: str) -> str:
        prompt = prompt_text.strip()
        if self.settings.normalize_whitespace:
            prompt = re.sub(r"\s+", " ", prompt)
        return prompt

    def _structural_validation(self, prompt: str, tokens: list[str]) -> ValidationStageResult:
        issues: list[str] = []

        if not prompt:
            issues.append("Please provide a business request before sending the prompt.")

        if len(prompt) < self.settings.min_prompt_length:
            issues.append("Please provide a more specific business request.")

        if len(prompt) > self.settings.max_prompt_length:
            issues.append("Prompt is too long for safe intake. Please shorten it or move large context to an attachment or retrieved source.")

        if len(tokens) < 3:
            issues.append("Please include enough task detail for the pipeline to act on.")

        return ValidationStageResult(
            stage_name="structural_validation",
            is_valid=not issues,
            issues=issues,
            details="Basic structure, length, and prompt-shape checks.",
        )

    def _security_validation(self, prompt: str, lowered: str) -> ValidationStageResult:
        issues: list[str] = []

        if self.settings.block_unsafe_requests:
            for pattern in UNSAFE_PATTERNS:
                if pattern in lowered:
                    issues.append(f"Unsafe request detected: '{pattern}'. Please rephrase the prompt safely.")

        if self.settings.block_sensitive_data:
            for pattern, message in SENSITIVE_DATA_PATTERNS:
                if pattern.search(prompt):
                    issues.append(message)

        return ValidationStageResult(
            stage_name="security_validation",
            is_valid=not issues,
            issues=issues,
            details="Policy-evasion and sensitive-data checks.",
        )

    def _domain_validation(self, pipeline_type: PipelineType, prompt: str, token_set: set[str]) -> ValidationStageResult:
        issues: list[str] = []

        if pipeline_type == PipelineType.RAG and self.settings.require_question_for_rag:
            has_rag_intent = "?" in prompt or bool(token_set.intersection(RAG_HINTS))
            if not has_rag_intent:
                issues.append("For a RAG pipeline, please state the knowledge question or retrieval intent you want answered.")

        if pipeline_type == PipelineType.AGENTIC_AI and self.settings.require_goal_for_agentic:
            has_agentic_intent = bool(token_set.intersection(AGENTIC_HINTS))
            if not has_agentic_intent:
                issues.append("For an Agentic pipeline, please describe the goal, workflow, or multi-step task clearly.")

        if pipeline_type == PipelineType.GENERATIVE_AI:
            has_generation_intent = bool(token_set.intersection(GENERATION_HINTS))
            if not has_generation_intent and len(token_set) < 6:
                issues.append("For an LLM pipeline, please describe the output you want generated, explained, or reviewed.")

        return ValidationStageResult(
            stage_name="domain_validation",
            is_valid=not issues,
            issues=issues,
            details="Pipeline-specific task intent and business-domain routing checks.",
        )

    def _fallback_semantic_validation(self, prompt: str, token_set: set[str]) -> ValidationStageResult:
        issues: list[str] = []

        if len(prompt.split()) < 6:
            issues.append("Semantic validation found the request too short to infer clear business intent.")

        if not token_set.intersection(SEMANTIC_CLARITY_HINTS):
            issues.append("Semantic validation suggests adding audience, format, or business context for clearer intent.")

        return ValidationStageResult(
            stage_name="semantic_validation",
            is_valid=not issues,
            issues=issues,
            details="Fallback semantic validation was used because LLM validation was unavailable.",
        )

    def _semantic_validation(self, prompt: str, token_set: set[str]) -> ValidationStageResult:
        if self.semantic_client is None:
            return self._fallback_semantic_validation(prompt, token_set)

        try:
            response = self.semantic_client.complete(SEMANTIC_VALIDATION_PROMPT.format(prompt_text=prompt))
            payload = json.loads(response.content)
            issues = payload.get("issues", [])
            return ValidationStageResult(
                stage_name="semantic_validation",
                is_valid=bool(payload.get("is_valid", not issues)),
                issues=[str(item) for item in issues],
                details=str(payload.get("details", "LLM-based semantic validation completed.")),
            )
        except Exception:
            return self._fallback_semantic_validation(prompt, token_set)

    def validate(self, pipeline_type: PipelineType, prompt_text: str) -> PromptValidationReport:
        prompt = self._normalize_prompt(prompt_text)
        lowered = prompt.lower()
        tokens = re.findall(r"[a-zA-Z0-9']+", lowered)
        token_set = set(tokens)

        stage_results = [
            self._structural_validation(prompt, tokens),
            self._security_validation(prompt, lowered),
            self._domain_validation(pipeline_type, prompt, token_set),
            self._semantic_validation(prompt, token_set),
        ]
        issues = [issue for stage in stage_results for issue in stage.issues]

        return PromptValidationReport(
            is_valid=not issues,
            issues=issues,
            normalized_prompt=prompt,
            validation_approach="four_stage_validation",
            stage_results=stage_results,
        )
