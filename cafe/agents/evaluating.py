from typing import Dict, Any, List
import pandas as pd
from .base_agent import BaseAgent
from cafe.core.snowflake_client import SnowflakeClient
from cafe.models.data_frame_builder import DataFrameBuilder
from cafe.strategies.evaluation import EvaluationStrategy, XGBoostEvaluation

class EvaluatingAgent(BaseAgent):
    """Agent that evaluates query results for ML model performance."""
    def __init__(self, snowflake_client: SnowflakeClient, evaluation_strategy: EvaluationStrategy = None):
        super().__init__()
        self.snowflake_client = snowflake_client
        self.evaluation_strategy = evaluation_strategy or XGBoostEvaluation()
        self.data_frame_builder = DataFrameBuilder()

    def run(self, queries: List[str]) -> Dict[str, Any]:
        """Build a DataFrame from query results and evaluate ML model."""
        data_frames = []
        for query in queries:
            result = self.snowflake_client.execute_query(query)
            df = self.data_frame_builder.build(result)
            data_frames.append(df)
        combined_df = pd.concat(data_frames, axis=1)
        metrics = self.evaluation_strategy.evaluate(combined_df)
        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics