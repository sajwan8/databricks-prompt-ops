from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

from src.databricks_prompt_ops.config import PromptOpsConfig, StorageBackend
from src.databricks_prompt_ops.models import PromptEvaluationReport, PromptRegistration, PromptTemplate


DEFAULT_PROMPT_TEMPLATES = [
    PromptTemplate(
        name="generic_clarification",
        version="1.0.0",
        description="Used to ask the user for more information when the prompt is incomplete.",
        template=(
            "I need more information before I can continue. Please provide:\n"
            "{issues}"
        ),
        tags=["clarification", "validation"],
    ),
    PromptTemplate(
        name="llm_response",
        version="1.0.0",
        description="Used for plain LLM-based generation after validation succeeds.",
        template=(
            "You are a helpful enterprise assistant.\n"
            "Respond to the user's request clearly and professionally.\n\n"
            "User request:\n{user_prompt}"
        ),
        tags=["llm", "generation"],
    ),
    PromptTemplate(
        name="rag_intake",
        version="1.0.0",
        description="Used to normalize and accept RAG requests before downstream retrieval.",
        template=(
            "RAG request accepted.\n"
            "User request:\n{user_prompt}\n\n"
            "Next step: send this validated prompt into retrieval and augmentation."
        ),
        tags=["rag", "intake"],
    ),
    PromptTemplate(
        name="agentic_intake",
        version="1.0.0",
        description="Used to normalize and accept agentic requests before downstream orchestration.",
        template=(
            "Agentic request accepted.\n"
            "User request:\n{user_prompt}\n\n"
            "Next step: send this validated prompt into the agent planner."
        ),
        tags=["agentic", "intake"],
    ),
]


class JsonListStore:
    """Simple JSON list store used for local development."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text("[]", encoding="utf-8")

    def load(self) -> list[dict]:
        return json.loads(self.file_path.read_text(encoding="utf-8"))

    def save(self, records: list[dict]) -> None:
        self.file_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def resolve_spark_session(prefer_databricks_connect: bool = True, spark_session=None):
    """Resolve a Spark session for Delta operations.

    Resolution order:
    1. Explicitly injected Spark session
    2. Active pyspark session (inside Databricks notebooks/jobs)
    3. Databricks Connect remote session
    4. Local Spark fallback when explicitly allowed by the environment
    """
    if spark_session is not None:
        return spark_session

    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise RuntimeError(
            "pyspark is required for Unity Catalog / Delta storage. "
            "Install pyspark and databricks-connect, or run inside Databricks."
        ) from exc

    active_session = SparkSession.getActiveSession()
    if active_session is not None:
        return active_session

    if prefer_databricks_connect:
        try:
            from databricks.connect import DatabricksSession
        except ImportError as exc:
            raise RuntimeError(
                "No active Spark session was found. For remote Unity Catalog access, "
                "install and configure databricks-connect, then create a session with "
                "`from databricks.connect import DatabricksSession` and "
                "`DatabricksSession.builder.getOrCreate()`, or inject that session into the pipeline."
            ) from exc

        try:
            return DatabricksSession.builder.getOrCreate()
        except Exception as exc:
            raise RuntimeError(
                "Failed to create a Databricks Connect Spark session. "
                "Make sure databricks-connect is installed and authenticated "
                "with your workspace, cluster or serverless compute."
            ) from exc

    try:
        return SparkSession.builder.getOrCreate()
    except Exception as exc:
        raise RuntimeError(
            "Unable to create a Spark session. If you are running outside a Databricks cluster, "
            "use Databricks Connect or pass an existing Spark session into the pipeline."
        ) from exc


class UnityCatalogDeltaStore:
    """Minimal Delta table wrapper for Unity Catalog-backed storage."""

    def __init__(
        self,
        catalog: str | None,
        schema: str | None,
        table_name: str,
        prefer_databricks_connect: bool = True,
        spark_session=None,
    ) -> None:
        if not catalog or not schema:
            raise ValueError("Unity Catalog storage requires both catalog and schema.")
        self.catalog = catalog
        self.schema = schema
        self.table_name = table_name
        self.prefer_databricks_connect = prefer_databricks_connect
        self._spark_session = spark_session

    @property
    def full_table_name(self) -> str:
        return f"`{self.catalog}`.`{self.schema}`.`{self.table_name}`"

    def _spark(self):
        self._spark_session = resolve_spark_session(
            prefer_databricks_connect=self.prefer_databricks_connect,
            spark_session=self._spark_session,
        )
        return self._spark_session

    def ensure_schema(self) -> None:
        spark = self._spark()
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{self.catalog}`.`{self.schema}`")


class PromptRegistryStore:
    def get(self, name: str) -> PromptTemplate:
        raise NotImplementedError


class JsonPromptRegistryStore(PromptRegistryStore):
    def __init__(self, file_path: str | Path) -> None:
        self.store = JsonListStore(file_path)
        self._seed_defaults()

    def _seed_defaults(self) -> None:
        if self.store.load():
            return
        self.store.save([asdict(item) for item in DEFAULT_PROMPT_TEMPLATES])

    def get(self, name: str) -> PromptTemplate:
        for item in self.store.load():
            if item["name"] == name:
                return PromptTemplate(**item)
        raise KeyError(f"Prompt template '{name}' not found.")


class DeltaPromptRegistryStore(PromptRegistryStore):
    def __init__(
        self,
        catalog: str | None,
        schema: str | None,
        table_name: str,
        prefer_databricks_connect: bool = True,
        spark_session=None,
    ) -> None:
        self.store = UnityCatalogDeltaStore(
            catalog=catalog,
            schema=schema,
            table_name=table_name,
            prefer_databricks_connect=prefer_databricks_connect,
            spark_session=spark_session,
        )
        self._ensure_table()
        self._seed_defaults()

    def _ensure_table(self) -> None:
        spark = self.store._spark()
        self.store.ensure_schema()
        spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {self.store.full_table_name} (
                name STRING,
                version STRING,
                description STRING,
                template STRING,
                tags ARRAY<STRING>
            )
            USING DELTA
            """
        )

    def _seed_defaults(self) -> None:
        spark = self.store._spark()
        if spark.table(self.store.full_table_name).limit(1).count() > 0:
            return
        rows = [asdict(item) for item in DEFAULT_PROMPT_TEMPLATES]
        spark.createDataFrame(rows).write.mode("append").saveAsTable(self.store.full_table_name)

    def get(self, name: str) -> PromptTemplate:
        from pyspark.sql import functions as F

        spark = self.store._spark()
        row = spark.table(self.store.full_table_name).where(F.col("name") == name).orderBy("version", ascending=False).first()
        if row is None:
            raise KeyError(f"Prompt template '{name}' not found.")
        return PromptTemplate(**row.asDict())


class PromptRequestStore:
    def register(self, prompt: PromptRegistration) -> None:
        raise NotImplementedError


class JsonPromptRequestStore(PromptRequestStore):
    def __init__(self, file_path: str | Path) -> None:
        self.store = JsonListStore(file_path)

    def register(self, prompt: PromptRegistration) -> None:
        records = self.store.load()
        records.append(asdict(prompt))
        self.store.save(records)


class DeltaPromptRequestStore(PromptRequestStore):
    def __init__(
        self,
        catalog: str | None,
        schema: str | None,
        table_name: str,
        prefer_databricks_connect: bool = True,
        spark_session=None,
    ) -> None:
        self.store = UnityCatalogDeltaStore(
            catalog=catalog,
            schema=schema,
            table_name=table_name,
            prefer_databricks_connect=prefer_databricks_connect,
            spark_session=spark_session,
        )
        self._ensure_table()

    def _ensure_table(self) -> None:
        spark = self.store._spark()
        self.store.ensure_schema()
        spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {self.store.full_table_name} (
                prompt_id STRING,
                user_id STRING,
                session_id STRING,
                pipeline_type STRING,
                prompt_text STRING,
                registered_at STRING,
                metadata MAP<STRING, STRING>
            )
            USING DELTA
            """
        )

    def register(self, prompt: PromptRegistration) -> None:
        spark = self.store._spark()
        record = asdict(prompt)
        record["metadata"] = {key: str(value) for key, value in record["metadata"].items()}
        spark.createDataFrame([record]).write.mode("append").saveAsTable(self.store.full_table_name)


class PromptEvaluationStore:
    def save(self, prompt_id: str, report: PromptEvaluationReport) -> None:
        raise NotImplementedError


class JsonPromptEvaluationStore(PromptEvaluationStore):
    def __init__(self, file_path: str | Path) -> None:
        self.store = JsonListStore(file_path)

    def save(self, prompt_id: str, report: PromptEvaluationReport) -> None:
        records = self.store.load()
        records.append(
            {
                "prompt_id": prompt_id,
                "evaluation": asdict(report),
            }
        )
        self.store.save(records)


class DeltaPromptEvaluationStore(PromptEvaluationStore):
    def __init__(
        self,
        catalog: str | None,
        schema: str | None,
        table_name: str,
        prefer_databricks_connect: bool = True,
        spark_session=None,
    ) -> None:
        self.store = UnityCatalogDeltaStore(
            catalog=catalog,
            schema=schema,
            table_name=table_name,
            prefer_databricks_connect=prefer_databricks_connect,
            spark_session=spark_session,
        )
        self._ensure_table()

    def _ensure_table(self) -> None:
        spark = self.store._spark()
        self.store.ensure_schema()
        spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {self.store.full_table_name} (
                prompt_id STRING,
                safety_score DOUBLE,
                reliability_score DOUBLE,
                fairness_score DOUBLE,
                overall_score DOUBLE,
                notes MAP<STRING, ARRAY<STRING>>
            )
            USING DELTA
            """
        )

    def save(self, prompt_id: str, report: PromptEvaluationReport) -> None:
        spark = self.store._spark()
        record = asdict(report)
        record["prompt_id"] = prompt_id
        spark.createDataFrame([record]).write.mode("append").saveAsTable(self.store.full_table_name)


def build_prompt_stores(
    config: PromptOpsConfig,
    spark_session=None,
) -> tuple[PromptRegistryStore, PromptRequestStore, PromptEvaluationStore]:
    if config.storage.backend == StorageBackend.DELTA:
        common_kwargs = {
            "prefer_databricks_connect": config.storage.prefer_databricks_connect,
            "spark_session": spark_session,
        }
        return (
            DeltaPromptRegistryStore(
                catalog=config.storage.catalog,
                schema=config.storage.schema,
                table_name=config.storage.registry_table,
                **common_kwargs,
            ),
            DeltaPromptRequestStore(
                catalog=config.storage.catalog,
                schema=config.storage.schema,
                table_name=config.storage.request_table,
                **common_kwargs,
            ),
            DeltaPromptEvaluationStore(
                catalog=config.storage.catalog,
                schema=config.storage.schema,
                table_name=config.storage.evaluation_table,
                **common_kwargs,
            ),
        )

    return (
        JsonPromptRegistryStore(Path(config.prompts.registry_path)),
        JsonPromptRequestStore(Path(config.prompts.request_store_path)),
        JsonPromptEvaluationStore(Path(config.prompts.evaluation_store_path)),
    )
