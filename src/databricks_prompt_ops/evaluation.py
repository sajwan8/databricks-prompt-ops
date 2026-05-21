from __future__ import annotations

import re

from src.databricks_prompt_ops.config import PipelineType
from src.databricks_prompt_ops.models import PromptEvaluationReport


BIAS_PATTERNS = [
    "people from",
    "because you are from",
    "women are",
    "men are",
]

UNSAFE_PATTERNS = [
    "bypass compliance",
    "avoid detection",
    "disable guardrails",
    "hide audit",
    "spoof identity",
]

CONSTRAINT_HINTS = {"timeline", "tone", "format", "audience", "length", "steps", "summary", "bullets", "table", "examples"}
RAG_HINTS = {"what", "how", "when", "where", "why", "which", "find", "lookup", "search", "retrieve"}
AGENTIC_HINTS = {"plan", "analyze", "investigate", "workflow", "steps", "goal", "orchestrate", "coordinate", "execute", "automate"}


class PromptEvaluator:
    """Evaluates prompts before downstream inference."""

    def evaluate(self, pipeline_type: PipelineType, prompt_text: str) -> PromptEvaluationReport:
        lowered = prompt_text.lower()
        tokens = re.findall(r"[a-zA-Z0-9']+", lowered)
        token_set = set(tokens)

        safety_score = 1.0
        reliability_score = 1.0
        fairness_score = 1.0

        safety_notes: list[str] = []
        reliability_notes: list[str] = []
        fairness_notes: list[str] = []

        if len(tokens) < 5:
            reliability_score -= 0.35
            reliability_notes.append("Prompt is very short and may not be actionable enough.")
        elif len(tokens) < 10:
            reliability_score -= 0.15
            reliability_notes.append("Prompt may benefit from clearer task constraints.")

        if not token_set.intersection(CONSTRAINT_HINTS):
            reliability_score -= 0.10
            reliability_notes.append("Prompt could be more reliable with format, audience, or output constraints.")

        if pipeline_type == PipelineType.RAG and "?" not in prompt_text and not token_set.intersection(RAG_HINTS):
            reliability_score -= 0.20
            reliability_notes.append("RAG prompts are more reliable when framed as a clear retrieval question.")

        if pipeline_type == PipelineType.AGENTIC_AI and not token_set.intersection(AGENTIC_HINTS):
            reliability_score -= 0.20
            reliability_notes.append("Agentic prompts work better when the goal or workflow is explicit.")

        if any(item in lowered for item in UNSAFE_PATTERNS):
            safety_score -= 0.60
            safety_notes.append("Prompt contains unsafe or policy-evasive intent.")

        if re.search(r"\b(?:\d[ -]*?){13,19}\b", prompt_text):
            safety_score -= 0.25
            safety_notes.append("Prompt appears to include sensitive numeric data that should be masked.")

        for pattern in BIAS_PATTERNS:
            if pattern in lowered:
                fairness_score -= 0.50
                fairness_notes.append(f"Potentially biased phrasing detected: '{pattern}'.")

        safety_score = max(round(safety_score, 3), 0.0)
        reliability_score = max(round(reliability_score, 3), 0.0)
        fairness_score = max(round(fairness_score, 3), 0.0)
        overall_score = round((safety_score + reliability_score + fairness_score) / 3, 3)

        return PromptEvaluationReport(
            safety_score=safety_score,
            reliability_score=reliability_score,
            fairness_score=fairness_score,
            overall_score=overall_score,
            notes={
                "safety": safety_notes or ["No major safety issues detected."],
                "reliability": reliability_notes or ["Prompt appears specific enough for execution."],
                "fairness": fairness_notes or ["No obvious fairness issues detected."],
            },
        )
