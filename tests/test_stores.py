from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.databricks_prompt_ops.config import StorageBackend
from src.databricks_prompt_ops.stores import build_prompt_stores, resolve_spark_session


class StoreSessionResolutionTests(unittest.TestCase):
    def test_explicit_session_is_returned_without_imports(self) -> None:
        sentinel = object()
        self.assertIs(resolve_spark_session(spark_session=sentinel), sentinel)

    def test_build_prompt_stores_tolerates_older_storage_settings(self) -> None:
        config = SimpleNamespace(
            storage=SimpleNamespace(
                backend=StorageBackend.DELTA,
                catalog="main",
                schema="prompt_ops",
                registry_table="prompt_registry",
                request_table="prompt_requests",
                evaluation_table="prompt_evaluations",
            ),
            prompts=SimpleNamespace(
                registry_path="unused",
                request_store_path="unused",
                evaluation_store_path="unused",
                template_directory=None,
            ),
        )

        with patch("src.databricks_prompt_ops.stores.MlflowPromptRegistryStore", return_value="registry"), patch(
            "src.databricks_prompt_ops.stores.DeltaPromptRequestStore",
            return_value="request_store",
        ), patch(
            "src.databricks_prompt_ops.stores.DeltaPromptEvaluationStore",
            return_value="evaluation_store",
        ):
            registry, request_store, evaluation_store = build_prompt_stores(config, spark_session=object())
        self.assertIsNotNone(registry)
        self.assertIsNotNone(request_store)
        self.assertIsNotNone(evaluation_store)


if __name__ == "__main__":
    unittest.main()
