from __future__ import annotations

import unittest

from src.databricks_prompt_ops.stores import resolve_spark_session


class StoreSessionResolutionTests(unittest.TestCase):
    def test_explicit_session_is_returned_without_imports(self) -> None:
        sentinel = object()
        self.assertIs(resolve_spark_session(spark_session=sentinel), sentinel)


if __name__ == "__main__":
    unittest.main()
