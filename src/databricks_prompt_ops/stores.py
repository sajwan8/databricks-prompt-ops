from __future__ import annotations

from dataclasses import asdict
from importlib import resources
from pathlib import Path
import json

from src.databricks_prompt_ops.config import PromptOpsConfig, StorageBackend
from src.databricks_prompt_ops.models import PromptEvaluationReport, PromptRegistration, PromptTemplate


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


class PromptTemplateRepository:
    """Loads prompt templates stored in Git-tracked YAML files."""

    def __init__(self, template_directory: str | None = None) -> None:
        self.template_directory = Path(template_directory) if template_directory else None

    def _parse_yaml_text(self, text: str) -> dict:
        try:
            import yaml  # type: ignore

            return yaml.safe_load(text)
        except ImportError:
            return self._fallback_parse_yaml_text(text)

    def _fallback_parse_yaml_text(self, text: str) -> dict:
        data: dict = {"tags": []}
        lines = text.splitlines()
        index = 0

        while index < len(lines):
            line = lines[index]
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                index += 1
                continue

            if stripped == "template: |":
                index += 1
                block_lines: list[str] = []
                while index < len(lines):
                    block_line = lines[index]
                    if block_line.startswith("  "):
                        block_lines.append(block_line[2:])
                        index += 1
                        continue
                    if block_line == "":
                        block_lines.append("")
                        index += 1
                        continue
                    break
                data["template"] = "\n".join(block_lines).rstrip("\n")
                continue

            if stripped == "tags:":
                index += 1
                tags: list[str] = []
                while index < len(lines):
                    tag_line = lines[index]
                    if tag_line.startswith("  - "):
                        tags.append(tag_line[4:].strip())
                        index += 1
                        continue
                    break
                data["tags"] = tags
                continue

            if ":" in line:
                key, value = line.split(":", 1)
                value = value.strip()
                data[key.strip()] = value.strip("'\"")
            index += 1

        return data

    def _load_template_from_path(self, path: Path) -> PromptTemplate:
        payload = self._parse_yaml_text(path.read_text(encoding="utf-8"))
        payload["source_path"] = path.as_posix()
        return PromptTemplate(**payload)

    def _load_template_from_resource(self, resource) -> PromptTemplate:
        payload = self._parse_yaml_text(resource.read_text(encoding="utf-8"))
        payload["source_path"] = f"package:{resource.name}"
        return PromptTemplate(**payload)

    def load_all(self) -> list[PromptTemplate]:
        if self.template_directory and self.template_directory.exists():
            return sorted(
                [self._load_template_from_path(path) for path in self.template_directory.glob("*.yaml")],
                key=lambda item: item.name,
            )

        template_package = resources.files("databricks_prompt_ops.prompt_management.templates")
        return sorted(
            [
                self._load_template_from_resource(resource)
                for resource in template_package.iterdir()
                if resource.name.endswith(".yaml")
            ],
            key=lambda item: item.name,
        )

    def get(self, name: str) -> PromptTemplate:
        for template in self.load_all():
            if template.name == name:
                return template
        raise KeyError(f"Prompt template '{name}' not found in the template repository.")


def resolve_spark_session(prefer_databricks_connect: bool = True, spark_session=None):
    """Resolve a Spark session for Delta operations."""
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
    def __init__(self, file_path: str | Path, template_repository: PromptTemplateRepository) -> None:
        self.store = JsonListStore(file_path)
        self.template_repository = template_repository
        self._seed_defaults()

    def _seed_defaults(self) -> None:
        records = [asdict(item) for item in self.template_repository.load_all()]
        self.store.save(records)

    def get(self, name: str) -> PromptTemplate:
        return self.template_repository.get(name)


class DeltaPromptRegistryStore(PromptRegistryStore):
    def __init__(
        self,
        catalog: str | None,
        schema: str | None,
        table_name: str,
        template_repository: PromptTemplateRepository,
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
        self.template_repository = template_repository
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
                source_path STRING,
                tags ARRAY<STRING>
            )
            USING DELTA
            """
        )

    def _seed_defaults(self) -> None:
        spark = self.store._spark()
        records = [asdict(item) for item in self.template_repository.load_all()]
        spark.createDataFrame(records).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
            self.store.full_table_name
        )

    def get(self, name: str) -> PromptTemplate:
        return self.template_repository.get(name)


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
                raw_user_input STRING,
                modified_prompt STRING,
                registered_at STRING,
                validation_approach STRING,
                inference_model_name STRING,
                prompt_template_name STRING,
                prompt_template_version STRING,
                prompt_template_source STRING,
                metadata MAP<STRING, STRING>
            )
            USING DELTA
            """
        )

    def register(self, prompt: PromptRegistration) -> None:
        spark = self.store._spark()
        record = asdict(prompt)
        record["metadata"] = {key: str(value) for key, value in record["metadata"].items()}
        spark.createDataFrame([record]).write.mode("append").option("mergeSchema", "true").saveAsTable(
            self.store.full_table_name
        )


class PromptEvaluationStore:
    def save(self, prompt_id: str, report: PromptEvaluationReport) -> None:
        raise NotImplementedError


class JsonPromptEvaluationStore(PromptEvaluationStore):
    def __init__(self, file_path: str | Path) -> None:
        self.store = JsonListStore(file_path)

    def save(self, prompt_id: str, report: PromptEvaluationReport) -> None:
        records = self.store.load()
        payload = asdict(report)
        payload["prompt_id"] = prompt_id
        records.append(payload)
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
                toxicity_score DOUBLE,
                correctness_score DOUBLE,
                completeness_score DOUBLE,
                consistency_score DOUBLE,
                relevance_score DOUBLE,
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
        spark.createDataFrame([record]).write.mode("append").option("mergeSchema", "true").saveAsTable(
            self.store.full_table_name
        )


def build_prompt_stores(
    config: PromptOpsConfig,
    spark_session=None,
) -> tuple[PromptRegistryStore, PromptRequestStore, PromptEvaluationStore]:
    template_repository = PromptTemplateRepository(getattr(config.prompts, "template_directory", None))

    if config.storage.backend == StorageBackend.DELTA:
        prefer_databricks_connect = getattr(config.storage, "prefer_databricks_connect", True)
        common_kwargs = {
            "prefer_databricks_connect": prefer_databricks_connect,
            "spark_session": spark_session,
        }
        return (
            DeltaPromptRegistryStore(
                catalog=config.storage.catalog,
                schema=config.storage.schema,
                table_name=config.storage.registry_table,
                template_repository=template_repository,
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
        JsonPromptRegistryStore(Path(config.prompts.registry_path), template_repository),
        JsonPromptRequestStore(Path(config.prompts.request_store_path)),
        JsonPromptEvaluationStore(Path(config.prompts.evaluation_store_path)),
    )
