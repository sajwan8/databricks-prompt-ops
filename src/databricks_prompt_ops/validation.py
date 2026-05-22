from __future__ import annotations

import re

from src.databricks_prompt_ops.config import PipelineType, ValidationSettings
from src.databricks_prompt_ops.models import PromptValidationReport


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


class PromptValidator:
    """Validates prompt completeness and basic safety before downstream execution."""

    def __init__(self, settings: ValidationSettings) -> None:
        self.settings = settings

    def _normalize_prompt(self, prompt_text: str) -> str:
        prompt = prompt_text.strip()
        if self.settings.normalize_whitespace:
            prompt = re.sub(r"\s+", " ", prompt)
        return prompt

    def validate(self, pipeline_type: PipelineType, prompt_text: str) -> PromptValidationReport:
        prompt = self._normalize_prompt(prompt_text)
        issues: list[str] = []
        lowered = prompt.lower()
        tokens = re.findall(r"[a-zA-Z0-9']+", lowered)
        token_set = set(tokens)

        if not prompt:
            issues.append("Please provide a business request before sending the prompt.")

        if len(prompt) < self.settings.min_prompt_length:
            issues.append("Please provide a more specific business request.")

        if len(prompt) > self.settings.max_prompt_length:
            issues.append("Prompt is too long for safe intake. Please shorten it or move large context to an attachment or retrieved source.")

        if len(tokens) < 3:
            issues.append("Please include enough task detail for the pipeline to act on.")

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
            if not has_generation_intent and len(tokens) < 6:
                issues.append("For an LLM pipeline, please describe the output you want generated, explained, or reviewed.")

        if self.settings.block_unsafe_requests:
            for pattern in UNSAFE_PATTERNS:
                if pattern in lowered:
                    issues.append(f"Unsafe request detected: '{pattern}'. Please rephrase the prompt safely.")

        if self.settings.block_sensitive_data:
            for pattern, message in SENSITIVE_DATA_PATTERNS:
                if pattern.search(prompt):
                    issues.append(message)

        return PromptValidationReport(
            is_valid=not issues,
            issues=issues,
            normalized_prompt=prompt,
            validation_approach=self.settings.validation_approach,
        )
