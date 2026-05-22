# Databricks Prompt Ops

This is a standalone partial LLMOps project focused only on prompt operations for Databricks-hosted applications.

It includes:
- Prompt ingestion
- Prompt registration and metadata storage
- Prompt validation
- Prompt evaluation for safety, reliability, fairness, toxicity, correctness, completeness, consistency, and relevance
- Configurable pipeline routing for:
  - `rag`
  - `agentic_ai`
  - `generative_ai`
- Interactive request processing for Databricks model serving style deployments
- Git-tracked YAML prompt templates
- Wheel-friendly `src/` packaging layout

## What This Module Does

This project does not implement full RAG retrieval or multi-agent orchestration.

Instead, it handles the prompt lifecycle before downstream execution:
1. User sends a prompt.
2. Prompt is validated and normalized.
3. The current prompt template is loaded from Git-tracked YAML files.
4. A `modified_prompt` is rendered from the normalized user input plus the selected template.
5. The modified prompt is evaluated before inference.
6. Prompt metadata is stored with:
   - `raw_user_input`
   - `modified_prompt`
   - `validation_approach`
   - `prompt_template_name`
   - `prompt_template_version`
   - `prompt_template_source`
   - `inference_model_name`
5. If validation fails, a clarification response is returned to the user.
6. If evaluation fails, downstream inference is blocked.
7. If validation and evaluation both succeed:
   - `generative_ai`: prompt is sent directly to the configured LLM
   - `rag`: prompt is accepted and routed as a RAG-ready request
   - `agentic_ai`: prompt is accepted and routed as an agentic-ready request

## Storage Backends

The pipeline now supports two storage modes:
- `json`: local development mode using files under `artifacts/`
- `delta`: Databricks mode using Unity Catalog Delta tables for prompt registry, request logs, and evaluation logs

Prompt templates are stored as YAML files under:
- `src/databricks_prompt_ops/prompt_management/templates/`

This makes the current prompt templates easy to version in GitHub and easy to package into the wheel.

`configs/prompt_pipeline_config.toml` is configured for Databricks Delta tables.

`configs/prompt_pipeline_config.local.toml` is configured for local JSON development and tests.

## Databricks Deployment Model

This project is designed so the main runtime can be wrapped in:
- Databricks notebook jobs
- Databricks Model Serving
- Databricks apps or internal APIs

The serving-style entrypoint is:
- `src/databricks_prompt_ops/serving.py`

## Quick Start

```bash
python main.py --config configs/prompt_pipeline_config.local.toml --user-id demo-user --session-id demo-session --message "claims help"
python main.py --config configs/prompt_pipeline_config.local.toml --user-id demo-user --session-id demo-session --message "Write a customer-ready response explaining the claims review timeline"
```

## Files

```text
databricks-prompt-ops/
  configs/
  artifacts/
  src/databricks_prompt_ops/
    prompt_management/templates/
  tests/
  main.py
  pyproject.toml
```

## Databricks Notes

Set `DATABRICKS_TOKEN` when using a real serving endpoint.

Configure the serving endpoint name in:
- `configs/prompt_pipeline_config.toml`

For local testing, the module falls back to a deterministic sample model.

### Unity Catalog Tables

When `storage.backend = "delta"`, the pipeline automatically creates these Delta tables if they do not already exist:
- `catalog.schema.prompt_registry`
- `catalog.schema.prompt_requests`
- `catalog.schema.prompt_evaluations`

The registry table is seeded automatically from the YAML prompt templates on first run.

When running outside a Databricks cluster, the Delta store now prefers a Databricks Connect remote Spark session when `storage.prefer_databricks_connect = true`.

### Live Databricks Integration

1. Create or choose a Unity Catalog catalog and schema for the pipeline, for example `main.prompt_ops`.
2. Update `configs/prompt_pipeline_config.toml` with the real catalog, schema, workspace URL, and serving endpoint.
3. Grant the serving identity or job cluster permissions:
   - `USE CATALOG`
   - `USE SCHEMA`
   - `CREATE TABLE`
   - `SELECT`
   - `INSERT`
4. Package `src/databricks_prompt_ops` with your Databricks job, app, or serving wrapper.
5. In notebooks or jobs, ensure a Spark session is available before calling the pipeline.
6. In Databricks Model Serving or app code, call `src.databricks_prompt_ops.serving.handle_request(...)`.
7. Point downstream RAG or agent orchestration only at prompts that return `status = "completed"`.

### Databricks Connect Usage

If you deploy this from your local machine, CI runner, or another non-cluster runtime, install and configure Databricks Connect first. The code now resolves Spark in this order:
1. Reuse an explicitly passed Spark session
2. Reuse an active Spark session
3. Create a Databricks Connect session
4. Fall back to plain local Spark only when Databricks Connect is not preferred

Recommended pattern:

```python
from databricks.connect import DatabricksSession

from src.databricks_prompt_ops.config import load_config
from src.databricks_prompt_ops.pipeline import DatabricksPromptOpsPipeline

spark = DatabricksSession.builder.getOrCreate()
config = load_config("configs/prompt_pipeline_config.toml")
pipeline = DatabricksPromptOpsPipeline(config, spark_session=spark)
```

You can do the same for the serving wrapper:

```python
from databricks.connect import DatabricksSession

from src.databricks_prompt_ops.serving import handle_request

spark = DatabricksSession.builder.getOrCreate()
response = handle_request(payload, config_path="configs/prompt_pipeline_config.toml", spark_session=spark)
```

If you want the pipeline to create the Databricks Connect session automatically, keep this in config:

```toml
[storage]
backend = "delta"
prefer_databricks_connect = true
```

Then make sure Databricks Connect authentication is already configured in your environment, typically through `.databrickscfg` or the standard Databricks environment variables before the process starts.

Example notebook usage:

```python
from src.databricks_prompt_ops.config import load_config
from src.databricks_prompt_ops.pipeline import DatabricksPromptOpsPipeline

config = load_config("configs/prompt_pipeline_config.toml")
pipeline = DatabricksPromptOpsPipeline(config)

result = pipeline.process_user_message(
    user_id="employee-123",
    session_id="session-456",
    prompt_text="Write a customer-ready explanation of the claims review timeline in bullet points.",
)

display(result.to_json())
```

Recommended production pattern:
- Keep this module as the intake and governance layer.
- Route `generative_ai` to Databricks Model Serving.
- Route `rag` only after `completed` status into retrieval, augmentation, and a second model call.
- Route `agentic_ai` only after `completed` status into your planner/executor framework.
- Monitor the Delta tables for prompt volume, validation failures, evaluation failures, and common clarification reasons.

## Packaging as a Wheel

The repo now includes `pyproject.toml` and keeps the prompt-management assets under the package tree so you can build a wheel cleanly.

Build locally:

```bash
python -m build
```

The resulting wheel will include:
- Python modules under `src/databricks_prompt_ops/`
- Git-tracked YAML prompt templates under `src/databricks_prompt_ops/prompt_management/templates/`

Recommended packaging pattern:
- Keep all prompt-management logic in this repo directory.
- Add any future prompt-management helpers under `src/databricks_prompt_ops/prompt_management/`.
- Avoid storing the current active templates only in notebooks or Delta tables; keep the source-of-truth YAML in Git.
