from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import uuid

from src.databricks_prompt_ops.config import PipelineType, PromptOpsConfig
from src.databricks_prompt_ops.evaluation import PromptEvaluator
from src.databricks_prompt_ops.logging_utils import get_logger
from src.databricks_prompt_ops.models import PromptOpsResponse, PromptRegistration, PromptValidationReport
from src.databricks_prompt_ops.serving_client import DatabricksServingClient, SampleModelClient
from src.databricks_prompt_ops.stores import PromptEvaluationStore, PromptRegistryStore, PromptRequestStore, build_prompt_stores
from src.databricks_prompt_ops.validation import PromptValidator


LOGGER = get_logger("databricks_prompt_ops.pipeline")


class DatabricksPromptOpsPipeline:
    """Interactive prompt operations pipeline for Databricks-hosted apps.

    This module is intentionally scoped only to prompt lifecycle management.
    It decides routing based on config and prompt type, but does not implement
    full RAG retrieval or full agent planning.
    """

    def __init__(self, config: PromptOpsConfig) -> None:
        self.config = config
        self.registry, self.request_store, self.evaluation_store = build_prompt_stores(config)
        self.validator = PromptValidator(config.validation)
        self.evaluator = PromptEvaluator()
        self.model_client = self._build_model_client()

    def _build_model_client(self) -> object:
        try:
            return DatabricksServingClient(
                workspace_url=self.config.databricks.workspace_url,
                serving_endpoint=self.config.databricks.serving_endpoint,
            )
        except Exception:
            return SampleModelClient()

    def _register_prompt(self, user_id: str, session_id: str, prompt_text: str) -> PromptRegistration:
        """Registers the raw user prompt and stores metadata for governance and debugging."""
        registration = PromptRegistration(
            prompt_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            pipeline_type=self.config.pipeline.type.value,
            prompt_text=prompt_text,
            registered_at=datetime.now(timezone.utc).isoformat(),
            metadata={
                "pipeline_name": self.config.pipeline.name,
                "environment": self.config.pipeline.environment,
            },
        )
        self.request_store.register(registration)
        return registration

    def _clarification_response(self, report: PromptValidationReport) -> str:
        """Formats the validation issues into a user-facing clarification message."""
        template = self.registry.get(self.config.prompts.clarification_prompt_name)
        issue_lines = "\n".join(f"- {issue}" for issue in report.issues)
        return template.template.format(issues=issue_lines)

    def _route_prompt(self, prompt_text: str) -> str:
        """Routes validated prompts according to the configured pipeline type."""
        pipeline_type = self.config.pipeline.type

        if pipeline_type == PipelineType.GENERATIVE_AI:
            template = self.registry.get(self.config.prompts.llm_prompt_name)
            rendered_prompt = template.template.format(user_prompt=prompt_text)
            try:
                return self.model_client.complete(rendered_prompt)
            except Exception as exc:
                LOGGER.warning("Databricks serving call failed, falling back to sample model: %s", exc)
                return SampleModelClient().complete(rendered_prompt)

        if pipeline_type == PipelineType.RAG:
            template = self.registry.get(self.config.prompts.rag_prompt_name)
            return template.template.format(user_prompt=prompt_text)

        template = self.registry.get(self.config.prompts.agentic_prompt_name)
        return template.template.format(user_prompt=prompt_text)

    def _evaluate_prompt_gate(self, prompt_id: str, prompt_text: str):
        if not self.config.evaluation.run_evaluation:
            return None

        evaluation_report = self.evaluator.evaluate(self.config.pipeline.type, prompt_text)
        self.evaluation_store.save(prompt_id, evaluation_report)

        fails_threshold = (
            evaluation_report.safety_score < self.config.evaluation.min_safety_score
            or evaluation_report.reliability_score < self.config.evaluation.min_reliability_score
            or evaluation_report.fairness_score < self.config.evaluation.min_fairness_score
            or evaluation_report.overall_score < self.config.evaluation.min_overall_score
        )
        return evaluation_report, fails_threshold

    def process_user_message(self, user_id: str, session_id: str, prompt_text: str) -> PromptOpsResponse:
        """Main interactive pipeline entrypoint.

        User sends a prompt.
        Raw prompt is registered.
        Prompt is validated and normalized.
        Normalized prompt is evaluated.
        If invalid, issues are sent back immediately.
        If valid and evaluation passes, the prompt is routed.
        """
        registration = self._register_prompt(user_id, session_id, prompt_text)
        validation_report = self.validator.validate(self.config.pipeline.type, prompt_text)

        if not validation_report.is_valid:
            return PromptOpsResponse(
                status="needs_clarification",
                pipeline_type=self.config.pipeline.type.value,
                prompt_id=registration.prompt_id,
                assistant_message=self._clarification_response(validation_report),
                registered_prompt=registration,
                validation_report=validation_report,
                evaluation_report=None,
                routed_downstream=False,
                llm_response=None,
            )

        evaluation_report = None
        evaluation_result = self._evaluate_prompt_gate(registration.prompt_id, validation_report.normalized_prompt)
        if evaluation_result is not None:
            evaluation_report, failed_evaluation = evaluation_result
            if failed_evaluation:
                return PromptOpsResponse(
                    status="failed_evaluation",
                    pipeline_type=self.config.pipeline.type.value,
                    prompt_id=registration.prompt_id,
                    assistant_message="Prompt evaluation found quality or policy issues. Please revise the request.",
                    registered_prompt=registration,
                    validation_report=validation_report,
                    evaluation_report=evaluation_report,
                    routed_downstream=False,
                    llm_response=None,
                )

        downstream_response = self._route_prompt(validation_report.normalized_prompt)
        llm_response = downstream_response if self.config.pipeline.type == PipelineType.GENERATIVE_AI else None

        if self.config.pipeline.type == PipelineType.GENERATIVE_AI:
            assistant_message = downstream_response
        elif self.config.pipeline.type == PipelineType.RAG:
            assistant_message = (
                "Prompt validated and registered successfully. "
                "This request is ready for the downstream RAG retrieval and augmentation pipeline.\n\n"
                f"{downstream_response}"
            )
        else:
            assistant_message = (
                "Prompt validated and registered successfully. "
                "This request is ready for the downstream agent planner.\n\n"
                f"{downstream_response}"
            )

        return PromptOpsResponse(
            status="completed",
            pipeline_type=self.config.pipeline.type.value,
            prompt_id=registration.prompt_id,
            assistant_message=assistant_message,
            registered_prompt=registration,
            validation_report=validation_report,
            evaluation_report=evaluation_report,
            routed_downstream=True,
            llm_response=llm_response,
        )
