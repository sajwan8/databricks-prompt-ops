from __future__ import annotations

import json
import os
from urllib import request

from src.databricks_prompt_ops.models import ModelCompletionResult


class SampleModelClient:
    """Deterministic local fallback for development and testing."""

    def __init__(self, model_name: str = "sample-local-model") -> None:
        self.model_name = model_name

    def complete(self, prompt_text: str) -> ModelCompletionResult:
        lowered = prompt_text.lower()
        if "return strict json with this schema" in lowered and "semantic quality only" in lowered:
            issues: list[str] = []
            if "worse managers" in lowered:
                issues.append("The request contains harmful or discriminatory framing that makes the intent unacceptable.")
            if "business prompt:" in lowered and len(prompt_text.split()) < 30:
                issues.append("The request would benefit from more business context or output constraints.")
            return ModelCompletionResult(
                content=json.dumps(
                    {
                        "is_valid": not issues,
                        "issues": issues,
                        "details": "Sample semantic validation completed.",
                    }
                ),
                model_name=self.model_name,
            )
        if "you are an evaluation judge for an enterprise prompt operations system" in lowered:
            payload = {
                "safety_score": 0.95,
                "reliability_score": 0.90,
                "fairness_score": 0.95,
                "toxicity_score": 0.98,
                "correctness_score": 0.90,
                "completeness_score": 0.88,
                "consistency_score": 0.90,
                "relevance_score": 0.92,
                "overall_score": 0.922,
                "notes": {
                    "safety": ["No major safety issues detected in the prompt/response pair."],
                    "reliability": ["Response is aligned to the requested task."],
                    "fairness": ["No obvious fairness concerns detected."],
                    "toxicity": ["No toxicity concerns detected."],
                    "correctness": ["Response appears generally correct for the supplied prompt."],
                    "completeness": ["Response covers the main requested points."],
                    "consistency": ["Response is reasonably consistent with the prompt intent."],
                    "relevance": ["Response is relevant to the business request."],
                },
            }
            if "worse managers" in lowered or "women are" in lowered or "men are" in lowered:
                payload["fairness_score"] = 0.20
                payload["safety_score"] = 0.45
                payload["overall_score"] = 0.65
                payload["notes"]["fairness"] = ["The prompt/response pair contains harmful or discriminatory framing."]
                payload["notes"]["safety"] = ["The prompt/response pair includes unsafe discriminatory content."]
            if "keywords or checks: timeline, claims" in lowered:
                payload["notes"]["relevance"].append("Requested keywords were considered during evaluation.")
            return ModelCompletionResult(content=json.dumps(payload), model_name=self.model_name)
        if "claims review timeline" in lowered:
            return ModelCompletionResult(
                content="The claims review timeline is 7 business days based on the request you provided.",
                model_name=self.model_name,
            )
        if "customer-ready response" in lowered:
            return ModelCompletionResult(
                content="Here is a professional customer-ready response drafted from your request.",
                model_name=self.model_name,
            )
        return ModelCompletionResult(
            content="Your validated prompt was accepted and processed successfully.",
            model_name=self.model_name,
        )


class DatabricksServingClient:
    """Calls a Databricks Model Serving endpoint.

    This is the client used when the module is hosted in Databricks or when a
    Databricks serving endpoint is exposed for prompt execution.
    """

    def __init__(
        self,
        workspace_url: str,
        serving_endpoint: str,
        configured_model_name: str | None = None,
        api_token: str | None = None,
    ) -> None:
        self.workspace_url = workspace_url.rstrip("/")
        self.serving_endpoint = serving_endpoint
        self.configured_model_name = configured_model_name or serving_endpoint
        self.api_token = api_token or os.getenv("DATABRICKS_TOKEN", "")

    def complete(self, prompt_text: str) -> ModelCompletionResult:
        if not self.api_token:
            raise ValueError("DATABRICKS_TOKEN is not set.")

        payload = json.dumps({"messages": [{"role": "user", "content": prompt_text}]}).encode("utf-8")
        http_request = request.Request(
            url=f"{self.workspace_url}/serving-endpoints/{self.serving_endpoint}/invocations",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_token}",
            },
            method="POST",
        )
        with request.urlopen(http_request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))

        if "choices" in body and body["choices"]:
            return ModelCompletionResult(
                content=body["choices"][0].get("message", {}).get("content", "").strip(),
                model_name=body.get("model", self.configured_model_name),
            )
        if "predictions" in body and body["predictions"]:
            first = body["predictions"][0]
            if isinstance(first, dict):
                return ModelCompletionResult(
                    content=str(first.get("content", first)).strip(),
                    model_name=str(first.get("model", self.configured_model_name)),
                )
            return ModelCompletionResult(
                content=str(first).strip(),
                model_name=self.configured_model_name,
            )
        return ModelCompletionResult(
            content=json.dumps(body),
            model_name=body.get("model", self.configured_model_name),
        )


class ResilientModelClient:
    """Uses the primary model client and falls back to the sample client on failure."""

    def __init__(self, primary_client, fallback_model_name: str) -> None:
        self.primary_client = primary_client
        self.fallback_client = SampleModelClient(fallback_model_name)

    def complete(self, prompt_text: str) -> ModelCompletionResult:
        try:
            return self.primary_client.complete(prompt_text)
        except Exception:
            return self.fallback_client.complete(prompt_text)
