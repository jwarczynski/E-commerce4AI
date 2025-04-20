from pathlib import Path

from cafe.agents import AgentFactory, EvaluatingAgent, FeatureEngineeringAgent, JudgeAgent
from cafe.core.semantic_model import SemanticModelManager
from cafe.core.snowflake_client import SnowflakeClient
from cafe.utils.logger import setup_logger


def main():
    logger = setup_logger(__name__)
    snowflake_client = SnowflakeClient()
    semantic_model_manager = SemanticModelManager()

    # Create agents
    feature_engineering_agent: FeatureEngineeringAgent = AgentFactory.create_agent(
        "feature_engineering", snowflake_client, semantic_model_manager
    )

    judge_agent: JudgeAgent = AgentFactory.create_agent("judge", snowflake_client)
    evaluating_agent: EvaluatingAgent = AgentFactory.create_agent("evaluating", snowflake_client)

    # Workflow
    semantic_model_path = Path.cwd() / "semantic_models" / "revenue_timeseries.yaml"

    # Step 1: Generate business question
    business_question = feature_engineering_agent.make_bussiness_quesiton(semantic_model_path=semantic_model_path)
    business_question = f"""{business_question}
     
Please provide the SQL query to achieve this. This query should either extend an existing database table by adding new columns while retaining the original ones, or potentially create an entirely new table."
"""
    # Step 2: Generate SQL query
    sql = feature_engineering_agent.run(business_question, semantic_model_path=semantic_model_path)

    # Step 3: Validate query
    validation_results = judge_agent.run(sql_query=sql, business_question=business_question)
    logger.debug(f"Validation results: {validation_results}")

    return
    if all(result["valid"] for result in validation_results.values()):
        # Step 3: Update semantic model
        semantic_model_manager.update_verified_queries(
            semantic_model_path,
            "daily_revenue_features",
            question=prompt,
            sql=sql,
        )

        # Step 4: Create new semantic model (example new table)
        new_table = {
            "name": "extended_daily_revenue",
            "description": "Extended table with 7-day moving average features",
            "base_table": {
                "database": "dog_cortex_analyst_demo",
                "schema": "revenue_timeseries",
                "table": "extended_daily_revenue"
            },
            "dimensions": [
                {"name": "date", "expr": "date", "data_type": "date"},
                {"name": "revenue_7day_avg", "expr": "revenue_7day_avg", "data_type": "number"}
            ]
        }
        new_model_path = "extended_revenue_timeseries.yaml"
        semantic_model_manager.create_new_semantic_model(semantic_model_path, new_model_path, new_table)

        # Step 5: Evaluate results
        metrics = evaluating_agent.run([sql])
        logger.info(f"Evaluation metrics: {metrics}")


if __name__ == "__main__":
    main()
