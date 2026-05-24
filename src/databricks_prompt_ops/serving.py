from __future__ import annotations

from src.databricks_prompt_ops.config import load_config
from src.databricks_prompt_ops.pipeline import DatabricksPromptOpsPipeline


def handle_request(payload: dict, config_path: str = "configs/prompt_pipeline_config.toml", spark_session=None) -> dict:
    """Databricks-friendly serving entrypoint.

    Expected payload:
    {
      "user_id": "...",
      "session_id": "...",
      "message": "..."
    }
    """
    config = load_config(config_path)
    pipeline = DatabricksPromptOpsPipeline(config, spark_session=spark_session)
    result = pipeline.process_user_message(
        user_id=payload["user_id"],
        session_id=payload["session_id"],
        prompt_text=payload["message"],
        evaluation_keywords=payload.get("evaluation_keywords"),
    )
    return {
        "status": result.status,
        "assistant_message": result.assistant_message,
        "pipeline_type": result.pipeline_type,
        "prompt_id": result.prompt_id,
        "routed_downstream": result.routed_downstream,
        "inference_model_name": result.inference_model_name,
    }
