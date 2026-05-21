# Databricks Prompt Ops

This is a standalone partial LLMOps project focused only on prompt operations for Databricks-hosted applications.

It includes:
- Prompt ingestion
- Prompt registration and metadata storage
- Prompt validation
- Prompt evaluation for safety, reliability, and fairness
- Configurable pipeline routing for:
  - `rag`
  - `agentic_ai`
  - `generative_ai`
- Interactive request processing for Databricks model serving style deployments

## What This Module Does

This project does not implement full RAG retrieval or multi-agent orchestration.

Instead, it handles the prompt lifecycle before downstream execution:
1. User sends a prompt.
2. Raw prompt is registered with metadata.
3. Prompt is validated and normalized.
4. Normalized prompt is evaluated for safety, reliability, and fairness.
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
  tests/
  main.py
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

The registry table is seeded automatically with the default prompt templates on first run.

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
