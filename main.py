from __future__ import annotations

import argparse

from src.databricks_prompt_ops.config import load_config
from src.databricks_prompt_ops.pipeline import DatabricksPromptOpsPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Databricks Prompt Ops pipeline")
    parser.add_argument("--config", required=True, help="Path to prompt pipeline TOML config")
    parser.add_argument("--user-id", required=True, help="User id for the request")
    parser.add_argument("--session-id", required=True, help="Session id for the request")
    parser.add_argument("--message", required=True, help="User prompt text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    #config = load_config(args.config)
    config = load_config("configs/prompt_pipeline_config.local.toml")
    pipeline = DatabricksPromptOpsPipeline(config)
    # result = pipeline.process_user_message(
    #     user_id=args.user_id,
    #     session_id=args.session_id,
    #     prompt_text=args.message,
    # )
    result = pipeline.process_user_message(
         user_id="demo-user",
         session_id="demo-session",
         prompt_text="Write a customer-ready response explaining the claims review timeline",
     )
    print(result.to_json())
# add model_names
#prompt history in git (explore gitlab)
#raw_user_input
#validation approach-dict("rule based validation", "elements_checked")
#toxicity, evaluation,]
#online repos check
#convert all into .whl

if __name__ == "__main__":
    main()
