from cafe.core.semantic_model import SemanticModelManager
from cafe.core.snowflake_client import SnowflakeClient

snowflake_client = SnowflakeClient()
semantic_model_manager = SemanticModelManager(snowflake_client)