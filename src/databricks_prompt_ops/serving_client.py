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
