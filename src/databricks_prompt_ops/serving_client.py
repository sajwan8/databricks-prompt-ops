from __future__ import annotations

import json
import os
from urllib import request


class SampleModelClient:
    """Deterministic local fallback for development and testing."""

    def complete(self, prompt_text: str) -> str:
        lowered = prompt_text.lower()
        if "claims review timeline" in lowered:
            return "The claims review timeline is 7 business days based on the request you provided."
        if "customer-ready response" in lowered:
            return "Here is a professional customer-ready response drafted from your request."
        return "Your validated prompt was accepted and processed successfully."


class DatabricksServingClient:
    """Calls a Databricks Model Serving endpoint.

    This is the client used when the module is hosted in Databricks or when a
    Databricks serving endpoint is exposed for prompt execution.
    """

    def __init__(self, workspace_url: str, serving_endpoint: str, api_token: str | None = None) -> None:
        self.workspace_url = workspace_url.rstrip("/")
        self.serving_endpoint = serving_endpoint
        self.api_token = api_token or os.getenv("DATABRICKS_TOKEN", "")

    def complete(self, prompt_text: str) -> str:
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
            return body["choices"][0].get("message", {}).get("content", "").strip()
        if "predictions" in body and body["predictions"]:
            first = body["predictions"][0]
            if isinstance(first, dict):
                return str(first.get("content", first)).strip()
            return str(first).strip()
        return json.dumps(body)
