from __future__ import annotations

from datetime import datetime, timezone
import uuid

from src.databricks_prompt_ops.config import PipelineType, PromptOpsConfig
from src.databricks_prompt_ops.evaluation import PromptEvaluator
from src.databricks_prompt_ops.logging_utils import get_logger
from src.databricks_prompt_ops.models import ModelCompletionResult, PromptOpsResponse, PromptRegistration, PromptTemplate, PromptValidationReport
from src.databricks_prompt_ops.serving_client import DatabricksServingClient, SampleModelClient
from src.databricks_prompt_ops.stores import build_prompt_stores
from src.databricks_prompt_ops.validation import PromptValidator


LOGGER = get_logger("databricks_prompt_ops.pipeline")


class DatabricksPromptOpsPipeline:
    """Interactive prompt operations pipeline for Databricks-hosted apps."""

    def __init__(self, config: PromptOpsConfig, spark_session=None) -> None:
        self.config = config
        self.spark_session = spark_session
        self.model_client = self._build_model_client()
        self.registry, self.request_store, self.evaluation_store = build_prompt_stores(
            config,
            spark_session=spark_session,
        )
        self.validator = PromptValidator(config.validation, semantic_client=self.model_client)
        self.evaluator = PromptEvaluator()

    def _build_model_client(self) -> object:
        try:
            return DatabricksServingClient(
                workspace_url=self.config.databricks.workspace_url,
                serving_endpoint=self.config.databricks.serving_endpoint,
                configured_model_name=self.config.models.llm_model,
            )
        except Exception:
            return SampleModelClient(self.config.models.fallback_model)

    def _clarification_response(self, report: PromptValidationReport) -> str:
        template = self.registry.get(self.config.prompts.clarification_prompt_name)
        issue_lines = "\n".join(f"- {issue}" for issue in report.issues)
        return template.template.format(issues=issue_lines)

    def _resolve_prompt_template(self) -> PromptTemplate:
        pipeline_type = self.config.pipeline.type
        if pipeline_type == PipelineType.GENERATIVE_AI:
            return self.registry.get(self.config.prompts.llm_prompt_name)
        if pipeline_type == PipelineType.RAG:
            return self.registry.get(self.config.prompts.rag_prompt_name)
        return self.registry.get(self.config.prompts.agentic_prompt_name)

    def _render_modified_prompt(self, template: PromptTemplate, normalized_prompt: str) -> str:
        return template.template.format(user_prompt=normalized_prompt)

    def _build_registration(
        self,
        user_id: str,
        session_id: str,
        raw_user_input: str,
        modified_prompt: str,
        validation_report: PromptValidationReport,
        prompt_template: PromptTemplate | None = None,
        inference_model_name: str | None = None,
        prompt_id: str | None = None,
    ) -> PromptRegistration:
        return PromptRegistration(
            prompt_id=prompt_id or str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            pipeline_type=self.config.pipeline.type.value,
            raw_user_input=raw_user_input,
            modified_prompt=modified_prompt,
            registered_at=datetime.now(timezone.utc).isoformat(),
            validation_approach=validation_report.validation_approach,
            inference_model_name=inference_model_name,
            prompt_template_name=prompt_template.name if prompt_template else None,
            prompt_template_version=prompt_template.version if prompt_template else None,
            prompt_template_source=prompt_template.source_path if prompt_template else None,
            metadata={
                "pipeline_name": self.config.pipeline.name,
                "environment": self.config.pipeline.environment,
            },
        )

    def _evaluate_prompt_gate(self, prompt_id: str, modified_prompt: str):
        if not self.config.evaluation.run_evaluation:
            return None

        evaluation_report = self.evaluator.evaluate(self.config.pipeline.type, modified_prompt)
        self.evaluation_store.save(prompt_id, evaluation_report)

        fails_threshold = (
            evaluation_report.safety_score < self.config.evaluation.min_safety_score
            or evaluation_report.reliability_score < self.config.evaluation.min_reliability_score
            or evaluation_report.fairness_score < self.config.evaluation.min_fairness_score
            or evaluation_report.overall_score < self.config.evaluation.min_overall_score
        )
        return evaluation_report, fails_threshold

    def _run_model_inference(self, modified_prompt: str) -> ModelCompletionResult:
        try:
            return self.model_client.complete(modified_prompt)
        except Exception as exc:
            LOGGER.warning("Databricks serving call failed, falling back to sample model: %s", exc)
            return SampleModelClient(self.config.models.fallback_model).complete(modified_prompt)

    def process_user_message(self, user_id: str, session_id: str, prompt_text: str) -> PromptOpsResponse:
        validation_report = self.validator.validate(self.config.pipeline.type, prompt_text)

        if not validation_report.is_valid:
            registration = self._build_registration(
                user_id=user_id,
                session_id=session_id,
                raw_user_input=prompt_text,
                modified_prompt=validation_report.normalized_prompt,
                validation_report=validation_report,
            )
            self.request_store.register(registration)
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
                inference_model_name=None,
            )

        prompt_template = self._resolve_prompt_template()
        modified_prompt = self._render_modified_prompt(prompt_template, validation_report.normalized_prompt)
        registration = self._build_registration(
            user_id=user_id,
            session_id=session_id,
            raw_user_input=prompt_text,
            modified_prompt=modified_prompt,
            validation_report=validation_report,
            prompt_template=prompt_template,
            prompt_id=None,
        )

        evaluation_report = None
        evaluation_result = self._evaluate_prompt_gate(registration.prompt_id, modified_prompt)
        if evaluation_result is not None:
            evaluation_report, failed_evaluation = evaluation_result
            if failed_evaluation:
                self.request_store.register(registration)
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
                    inference_model_name=None,
                )

        inference_model_name = None
        llm_response = None
        if self.config.pipeline.type == PipelineType.GENERATIVE_AI:
            inference_result = self._run_model_inference(modified_prompt)
            inference_model_name = inference_result.model_name
            llm_response = inference_result.content
            assistant_message = inference_result.content
        elif self.config.pipeline.type == PipelineType.RAG:
            assistant_message = (
                "Prompt validated and registered successfully. "
                "This request is ready for the downstream RAG retrieval and augmentation pipeline.\n\n"
                f"{modified_prompt}"
            )
        else:
            assistant_message = (
                "Prompt validated and registered successfully. "
                "This request is ready for the downstream agent planner.\n\n"
                f"{modified_prompt}"
            )

        registration = self._build_registration(
            user_id=user_id,
            session_id=session_id,
            raw_user_input=prompt_text,
            modified_prompt=modified_prompt,
            validation_report=validation_report,
            prompt_template=prompt_template,
            inference_model_name=inference_model_name,
            prompt_id=registration.prompt_id,
        )
        self.request_store.register(registration)

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
            inference_model_name=inference_model_name,
        )
