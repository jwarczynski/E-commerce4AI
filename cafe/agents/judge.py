from typing import Dict, Any
from .base_agent import BaseAgent
from cafe.core.snowflake_client import SnowflakeClient
from cafe.strategies.validation import ValidationStrategy, SyntaxValidation, ExecutionValidation, SemanticValidation

class JudgeAgent(BaseAgent):
    """Agent that validates SQL queries for correctness and usefulness."""
    def __init__(self, snowflake_client: SnowflakeClient, validation_strategies: list[ValidationStrategy] = None):
        super().__init__()
        self.snowflake_client = snowflake_client
        self.validation_strategies = validation_strategies or [
            SyntaxValidation(),
            ExecutionValidation(self.snowflake_client),
            SemanticValidation()
        ]

    def validate(self, sql: str, prompt: str, semantic_model: str) -> Dict[str, Any]:
        return self.run(sql, prompt, semantic_model)

    def run(self, sql: str, prompt: str, semantic_model: str) -> Dict[str, Any]:
        """Validate a SQL query using multiple strategies."""
        query_result = self.snowflake_client.execute_query(sql)

        results = {}
        for strategy in self.validation_strategies:
            is_valid, message = strategy.validate(sql, prompt, semantic_model, query_result)
            results[strategy.__class__.__name__] = {"valid": is_valid, "message": message}
            if not is_valid:
                self.logger.error(f"Validation failed: {message}")
                return results
        self.logger.info("All validations passed")
        return results