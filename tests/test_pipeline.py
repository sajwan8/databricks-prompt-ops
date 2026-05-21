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
                f"""
                [pipeline]
                name = "databricks-prompt-ops"
                type = "generative_ai"
                environment = "test"

                [models]
                llm_model = "databricks-meta-llama-3-3-70b-instruct"
                fallback_model = "sample-local-model"

                [prompts]
                registry_path = "{(workspace / "prompt_registry.json").as_posix()}"
                request_store_path = "{(workspace / "prompt_requests.json").as_posix()}"
                evaluation_store_path = "{(workspace / "prompt_evaluations.json").as_posix()}"
                clarification_prompt_name = "generic_clarification"
                llm_prompt_name = "llm_response"
                rag_prompt_name = "rag_intake"
                agentic_prompt_name = "agentic_intake"

                [storage]
                backend = "json"

                [databricks]
                workspace_url = "https://example.cloud.databricks.com"
                serving_endpoint = "unit-test-endpoint"

                [validation]
                min_prompt_length = 15
                require_question_for_rag = true
                require_goal_for_agentic = true
                block_unsafe_requests = true
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
            ).strip(),
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

    def test_valid_prompt_returns_response(self) -> None:
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

    def test_prompt_is_normalized_before_evaluation_and_routing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = self._build_pipeline(Path(tmp_dir))
            result = pipeline.process_user_message(
                user_id="user-5",
                session_id="session-5",
                prompt_text="  Write   a claims summary   for the customer in bullet format.  ",
            )
            self.assertEqual(result.validation_report.normalized_prompt, "Write a claims summary for the customer in bullet format.")


if __name__ == "__main__":
    unittest.main()
