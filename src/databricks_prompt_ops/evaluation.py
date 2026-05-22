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

TOXICITY_PATTERNS = ["stupid", "idiot", "hate", "worthless", "kill", "destroy them"]
CONSTRAINT_HINTS = {"timeline", "tone", "format", "audience", "length", "steps", "summary", "bullets", "table", "examples"}
RAG_HINTS = {"what", "how", "when", "where", "why", "which", "find", "lookup", "search", "retrieve"}
AGENTIC_HINTS = {"plan", "analyze", "investigate", "workflow", "steps", "goal", "orchestrate", "coordinate", "execute", "automate"}
CORRECTNESS_HINTS = {"source", "cite", "reference", "evidence", "accurate", "fact"}
COMPLETENESS_HINTS = {"include", "cover", "all", "end-to-end", "comprehensive", "detailed"}
CONSISTENCY_HINTS = {"same", "consistent", "standard", "policy", "template"}
RELEVANCE_HINTS = {"customer", "claims", "request", "business", "context", "timeline", "response"}


class PromptEvaluator:
    """Evaluates prompts before downstream inference."""

    def evaluate(self, pipeline_type: PipelineType, prompt_text: str) -> PromptEvaluationReport:
        lowered = prompt_text.lower()
        tokens = re.findall(r"[a-zA-Z0-9']+", lowered)
        token_set = set(tokens)

        safety_score = 1.0
        reliability_score = 1.0
        fairness_score = 1.0
        toxicity_score = 1.0
        correctness_score = 1.0
        completeness_score = 1.0
        consistency_score = 1.0
        relevance_score = 1.0

        safety_notes: list[str] = []
        reliability_notes: list[str] = []
        fairness_notes: list[str] = []
        toxicity_notes: list[str] = []
        correctness_notes: list[str] = []
        completeness_notes: list[str] = []
        consistency_notes: list[str] = []
        relevance_notes: list[str] = []

        if len(tokens) < 5:
            reliability_score -= 0.35
            completeness_score -= 0.20
            reliability_notes.append("Prompt is very short and may not be actionable enough.")
            completeness_notes.append("Prompt is too short to express the full task.")
        elif len(tokens) < 10:
            reliability_score -= 0.15
            completeness_score -= 0.10
            reliability_notes.append("Prompt may benefit from clearer task constraints.")

        if not token_set.intersection(CONSTRAINT_HINTS):
            reliability_score -= 0.10
            completeness_score -= 0.10
            reliability_notes.append("Prompt could be more reliable with format, audience, or output constraints.")
            completeness_notes.append("Prompt would be more complete with explicit output constraints.")

        if pipeline_type == PipelineType.RAG and "?" not in prompt_text and not token_set.intersection(RAG_HINTS):
            reliability_score -= 0.20
            relevance_score -= 0.15
            reliability_notes.append("RAG prompts are more reliable when framed as a clear retrieval question.")
            relevance_notes.append("RAG prompt lacks clear retrieval intent.")

        if pipeline_type == PipelineType.AGENTIC_AI and not token_set.intersection(AGENTIC_HINTS):
            reliability_score -= 0.20
            consistency_score -= 0.15
            reliability_notes.append("Agentic prompts work better when the goal or workflow is explicit.")
            consistency_notes.append("Agentic prompt lacks a stable workflow or goal description.")

        if any(item in lowered for item in UNSAFE_PATTERNS):
            safety_score -= 0.60
            toxicity_score -= 0.30
            safety_notes.append("Prompt contains unsafe or policy-evasive intent.")
            toxicity_notes.append("Prompt contains hostile or unsafe operational intent.")

        if re.search(r"\b(?:\d[ -]*?){13,19}\b", prompt_text):
            safety_score -= 0.25
            correctness_score -= 0.10
            safety_notes.append("Prompt appears to include sensitive numeric data that should be masked.")
            correctness_notes.append("Sensitive data in the prompt may reduce safe and correct handling.")

        for pattern in BIAS_PATTERNS:
            if pattern in lowered:
                fairness_score -= 0.50
                relevance_score -= 0.10
                fairness_notes.append(f"Potentially biased phrasing detected: '{pattern}'.")
                relevance_notes.append("Biased framing may distract from the real business task.")

        if any(pattern in lowered for pattern in TOXICITY_PATTERNS):
            toxicity_score -= 0.50
            toxicity_notes.append("Prompt includes toxic or abusive phrasing.")

        if not token_set.intersection(CORRECTNESS_HINTS):
            correctness_score -= 0.10
            correctness_notes.append("Prompt does not specify any factuality, evidence, or accuracy expectations.")

        if not token_set.intersection(COMPLETENESS_HINTS):
            completeness_score -= 0.10
            completeness_notes.append("Prompt could better define coverage or completeness expectations.")

        if not token_set.intersection(CONSISTENCY_HINTS):
            consistency_score -= 0.10
            consistency_notes.append("Prompt does not specify consistency with policy, standard, or prior format.")

        if not token_set.intersection(RELEVANCE_HINTS):
            relevance_score -= 0.15
            relevance_notes.append("Prompt could be more grounded in the business context or target use case.")

        scores = {
            "safety_score": max(round(safety_score, 3), 0.0),
            "reliability_score": max(round(reliability_score, 3), 0.0),
            "fairness_score": max(round(fairness_score, 3), 0.0),
            "toxicity_score": max(round(toxicity_score, 3), 0.0),
            "correctness_score": max(round(correctness_score, 3), 0.0),
            "completeness_score": max(round(completeness_score, 3), 0.0),
            "consistency_score": max(round(consistency_score, 3), 0.0),
            "relevance_score": max(round(relevance_score, 3), 0.0),
        }
        overall_score = round(sum(scores.values()) / len(scores), 3)

        return PromptEvaluationReport(
            overall_score=overall_score,
            notes={
                "safety": safety_notes or ["No major safety issues detected."],
                "reliability": reliability_notes or ["Prompt appears specific enough for execution."],
                "fairness": fairness_notes or ["No obvious fairness issues detected."],
                "toxicity": toxicity_notes or ["No toxicity concerns detected."],
                "correctness": correctness_notes or ["Prompt includes enough cues for fact-aware handling."],
                "completeness": completeness_notes or ["Prompt scope appears sufficiently complete."],
                "consistency": consistency_notes or ["Prompt should produce reasonably consistent handling."],
                "relevance": relevance_notes or ["Prompt appears relevant to the stated business task."],
            },
            **scores,
        )
