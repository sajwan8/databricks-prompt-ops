from __future__ import annotations

from pathlib import Path
import tempfile
import textwrap
import unittest

from src.databricks_prompt_ops.config import load_config
from src.databricks_prompt_ops.pipeline import DatabricksPromptOpsPipeline


class PromptOpsPipelineTests(unittest.TestCase):
    def _build_config(self, workspace: Path) -> Path:
        config_path = workspace / "prompt_pipeline_config.toml"
        config_path.write_text(
            textwrap.dedent(
                """
                [pipeline]
                name = "databricks-prompt-ops"
                type = "generative_ai"
                environment = "test"

                [models]
                llm_model = "gpt-4.1"
                fallback_model = "sample-local-model"

                [prompts]
                registry_path = "REGISTRY_STORE_PATH"
                request_store_path = "REQUEST_STORE_PATH"
                evaluation_store_path = "EVALUATION_STORE_PATH"
                template_directory = "src/databricks_prompt_ops/prompt_management/templates"
                clarification_prompt_name = "generic_clarification"
                llm_prompt_name = "llm_response"
                rag_prompt_name = "rag_intake"
                agentic_prompt_name = "agentic_intake"

                [storage]
                backend = "json"
                prefer_databricks_connect = false

                [mlflow]
                tracking_uri = ""
                registry_uri = ""
                prompt_alias = "latest"
                sync_prompts_on_startup = true

                [databricks]
                workspace_url = "https://example.cloud.databricks.com"
                serving_endpoint = "unit-test-endpoint"

                [validation]
                min_prompt_length = 15
                require_question_for_rag = true
                require_goal_for_agentic = true
                block_unsafe_requests = true
                validation_approach = "four_stage_validation"
                max_prompt_length = 4000
                block_sensitive_data = true
                normalize_whitespace = true

                [evaluation]
                run_evaluation = true
                min_safety_score = 0.70
                min_reliability_score = 0.60
                min_fairness_score = 0.70
                min_overall_score = 0.70
                """
            )
            .strip()
            .replace("REGISTRY_STORE_PATH", (workspace / "prompt_registry.json").as_posix())
            .replace("REQUEST_STORE_PATH", (workspace / "prompt_requests.json").as_posix())
            .replace("EVALUATION_STORE_PATH", (workspace / "prompt_evaluations.json").as_posix()),
            encoding="utf-8",
        )
        return config_path

    def _build_pipeline(self, workspace: Path) -> DatabricksPromptOpsPipeline:
        return DatabricksPromptOpsPipeline(load_config(self._build_config(workspace)))

    def test_invalid_prompt_returns_clarification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._build_pipeline(Path(tmp_dir))
            result = pipeline.process_user_message(
                user_id="user-1",
                session_id="session-1",
                prompt_text="claims",
            )
            self.assertEqual(result.status, "needs_clarification")
            self.assertFalse(result.validation_report.is_valid)
            self.assertEqual(result.validation_report.validation_approach, "four_stage_validation")
            self.assertEqual(
                [stage.stage_name for stage in result.validation_report.stage_results],
                [
                    "structural_validation",
                    "security_validation",
                    "domain_validation",
                    "semantic_validation",
                ],
            )

    def test_valid_prompt_returns_response_with_model_and_prompt_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._build_pipeline(Path(tmp_dir))
            result = pipeline.process_user_message(
                user_id="user-2",
                session_id="session-2",
                prompt_text="Write a customer-ready response explaining the claims review timeline in bullet points.",
            )
            self.assertEqual(result.status, "completed")
            self.assertTrue(result.validation_report.is_valid)
            self.assertTrue(result.routed_downstream)
            self.assertEqual(result.inference_model_name, "sample-local-model")
            self.assertEqual(result.registered_prompt.raw_user_input, "Write a customer-ready response explaining the claims review timeline in bullet points.")
            self.assertIn("You are a helpful enterprise assistant.", result.registered_prompt.modified_prompt)
            self.assertEqual(result.registered_prompt.validation_approach, "four_stage_validation")
            self.assertEqual(result.registered_prompt.prompt_template_name, "llm_response")
            self.assertTrue(result.registered_prompt.prompt_template_source.endswith("llm_response.yaml"))

    def test_validation_report_contains_four_stage_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._build_pipeline(Path(tmp_dir))
            result = pipeline.process_user_message(
                user_id="user-stages",
                session_id="session-stages",
                prompt_text="Write a customer-ready response explaining the claims review timeline in bullet points.",
            )
            self.assertEqual(len(result.validation_report.stage_results), 4)
            self.assertEqual(result.validation_report.stage_results[0].stage_name, "structural_validation")
            self.assertEqual(result.validation_report.stage_results[1].stage_name, "security_validation")
            self.assertEqual(result.validation_report.stage_results[2].stage_name, "domain_validation")
            self.assertEqual(result.validation_report.stage_results[3].stage_name, "semantic_validation")

    def test_sensitive_data_prompt_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._build_pipeline(Path(tmp_dir))
            result = pipeline.process_user_message(
                user_id="user-3",
                session_id="session-3",
                prompt_text="Write a response to john.doe@example.com and mention card 4111 1111 1111 1111 in the summary.",
            )
            self.assertEqual(result.status, "needs_clarification")
            self.assertIn("Please remove payment card numbers", " ".join(result.validation_report.issues))

    def test_biased_prompt_fails_evaluation_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._build_pipeline(Path(tmp_dir))
            result = pipeline.process_user_message(
                user_id="user-4",
                session_id="session-4",
                prompt_text="Write a management brief explaining why women are worse managers in bullet points for executives.",
            )
            self.assertEqual(result.status, "failed_evaluation")
            self.assertFalse(result.routed_downstream)
            self.assertIsNotNone(result.evaluation_report)
            self.assertLess(result.evaluation_report.fairness_score, 0.70)

    def test_prompt_is_normalized_before_templating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._build_pipeline(Path(tmp_dir))
            result = pipeline.process_user_message(
                user_id="user-5",
                session_id="session-5",
                prompt_text="  Write   a claims summary   for the customer in bullet format.  ",
            )
            self.assertEqual(result.validation_report.normalized_prompt, "Write a claims summary for the customer in bullet format.")
            self.assertIn("Write a claims summary for the customer in bullet format.", result.registered_prompt.modified_prompt)

    def test_evaluation_report_contains_additional_quality_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._build_pipeline(Path(tmp_dir))
            result = pipeline.process_user_message(
                user_id="user-6",
                session_id="session-6",
                prompt_text="Write a customer response about the claims timeline with accurate bullet points and include all important details.",
            )
            self.assertIsNotNone(result.evaluation_report)
            self.assertTrue(hasattr(result.evaluation_report, "toxicity_score"))
            self.assertTrue(hasattr(result.evaluation_report, "correctness_score"))
            self.assertTrue(hasattr(result.evaluation_report, "completeness_score"))
            self.assertTrue(hasattr(result.evaluation_report, "consistency_score"))
            self.assertTrue(hasattr(result.evaluation_report, "relevance_score"))


if __name__ == "__main__":
    unittest.main()
