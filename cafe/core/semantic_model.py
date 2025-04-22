import os
import time
from pathlib import Path
from typing import Any, Dict

import yaml
from snowflake.connector.errors import ProgrammingError

from .snowflake_client import SnowflakeClient
from ..models.semantic_model_graph import SemanticModelGraph
from ..utils.logger import setup_logger


class SemanticModelManager:
    """Manages semantic models, including loading, updating, and graph tracking with Snowflake stage integration.

    Attributes:
        logger: Configured logger instance for the class.
        graph: SemanticModelGraph instance for tracking model relationships.
        snowflake_client: SnowflakeClient instance for Snowflake operations.
        stage_name: Name of the Snowflake stage for storing semantic models.
    """

    def __init__(self, snowflake_client: SnowflakeClient, stage_name: str = "SEMANTIC_MODELS_STAGE"):
        """Initialize SemanticModelManager with a SnowflakeClient instance.

        Args:
            snowflake_client: Initialized SnowflakeClient instance for Snowflake operations.
            stage_name: Name of the Snowflake stage to use for storing semantic models.

        Raises:
            ValueError: If snowflake_client is not properly initialized.
        """
        self.logger = setup_logger(__name__)
        self.graph = SemanticModelGraph()
        self.stage_name = stage_name
        if not isinstance(snowflake_client, SnowflakeClient):
            raise ValueError("snowflake_client must be an instance of SnowflakeClient")
        self.snowflake_client = snowflake_client
        self._ensure_stage_exists()

    def _ensure_stage_exists(self) -> None:
        """Ensure the Snowflake stage exists, creating it if necessary.

        Raises:
            ProgrammingError: If stage creation or verification fails due to permissions or other issues.
        """
        try:
            query = f"CREATE STAGE IF NOT EXISTS {self.stage_name}"
            self.snowflake_client.execute_query(query)
            self.logger.info(f"Verified/Created Snowflake stage: {self.stage_name}")
        except ProgrammingError as e:
            self.logger.error(f"Failed to create/verify Snowflake stage {self.stage_name}: {str(e)}")
            raise

    def load_yaml(self, file_path: str | Path) -> str:
        """Load a YAML file as a string from local filesystem.

        Args:
            file_path: Path to the YAML file.

        Returns:
            str: Content of the YAML file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Semantic model file not found: {file_path}")
        with open(file_path, 'r') as file:
            return file.read()

    def parse_yaml(self, yaml_content: str) -> Dict[str, Any]:
        """Parse YAML content into a dictionary.

        Args:
            yaml_content: YAML content as a string.

        Returns:
            Dict[str, Any]: Parsed YAML content.

        Raises:
            yaml.YAMLError: If YAML parsing fails.
        """
        try:
            return yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            self.logger.error(f"Failed to parse YAML content: {str(e)}")
            raise

    def load_yaml_from_stage(self, file_name: str) -> str:
        """Load a YAML file from Snowflake stage.

        Args:
            file_name: Name of the file in the Snowflake stage.

        Returns:
            str: Content of the YAML file.

        Raises:
            ProgrammingError: If file retrieval from stage fails.
        """
        try:
            # Download file from stage to temporary location
            query = f"GET @{self.stage_name}/{file_name} 'file:///tmp/{file_name}'"
            self.snowflake_client.execute_query(query)
            with open(f"/tmp/{file_name}", 'r') as file:
                content = file.read()
            os.remove(f"/tmp/{file_name}")
            self.logger.info(f"Loaded {file_name} from Snowflake stage {self.stage_name}")
            return content
        except ProgrammingError as e:
            self.logger.error(f"Failed to load {file_name} from Snowflake stage {self.stage_name}: {str(e)}")
            raise

    def update_verified_queries(
            self,
            file_path: str | Path,
            query_name: str,
            question: str,
            sql: str,
            verified_by: str = "system"
    ) -> None:
        """Update the verified_queries section of a semantic model in both local filesystem and Snowflake stage.

        Args:
            file_path: Path to the semantic model file.
            query_name: Name of the query.
            question: Question associated with the query.
            sql: SQL statement for the query.
            verified_by: Entity that verified the query.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ProgrammingError: If Snowflake stage update fails.
            yaml.YAMLError: If YAML parsing or writing fails.
        """
        file_path = Path(file_path)
        file_name = file_path.name

        # Update local file
        yaml_content = self.load_yaml(file_path)
        model_dict = self.parse_yaml(yaml_content)
        if "verified_queries" not in model_dict:
            model_dict["verified_queries"] = []
        model_dict["verified_queries"].append(
            {
                "name": query_name,
                "question": question,
                "sql": sql,
                "verified_at": int(time.time()),
                "verified_by": verified_by
            }
        )
        with open(file_path, 'w') as file:
            yaml.safe_dump(model_dict, file)
        self.logger.info(f"Updated verified_queries in local file {file_path}")

        # Update Snowflake stage
        try:
            query = f"PUT file://{file_path} @{self.stage_name} OVERWRITE = TRUE"
            self.snowflake_client.execute_query(query)
            self.logger.info(f"Updated {file_name} in Snowflake stage {self.stage_name}")
        except ProgrammingError as e:
            self.logger.error(f"Failed to update {file_name} in Snowflake stage {self.stage_name}: {str(e)}")
            raise

    def add_new_semantic_model(
            self,
            semantic_model: str,
            new_model_path: str | Path,
            base_model_path: str | Path = None
    ) -> None:
        """Add a new semantic model to the graph, local filesystem, and Snowflake stage.

        Args:
            semantic_model: YAML content of the new semantic model.
            new_model_path: Path where the new model will be saved.
            base_model_path: Optional path to the base model for graph tracking.

        Raises:
            OSError: If file writing fails.
            ProgrammingError: If Snowflake stage upload fails.
        """
        new_model_path = Path(new_model_path)
        file_name = new_model_path.name

        # Save to local filesystem
        with open(new_model_path, 'w') as file:
            file.write(semantic_model)
        self.logger.info(f"Saved new semantic model to local file {new_model_path}")

        # Update graph
        self.graph.add_edge(base_model_path, new_model_path)

        # Upload to Snowflake stage
        try:
            query = f"PUT file://{new_model_path} @{self.stage_name} OVERWRITE = TRUE"
            self.snowflake_client.execute_query(query)
            self.logger.info(f"Uploaded new semantic model {file_name} to Snowflake stage {self.stage_name}")
        except ProgrammingError as e:
            self.logger.error(f"Failed to upload {file_name} to Snowflake stage {self.stage_name}: {str(e)}")
            raise

    def show_semantic_model_graph(self) -> None:
        """Display the semantic model graph.

        Raises:
            ValueError: If graph display fails.
        """
        try:
            self.graph.display_graph()
        except Exception as e:
            self.logger.error(f"Failed to display semantic model graph: {str(e)}")
            raise ValueError(f"Graph display failed: {str(e)}")

    def __del__(self):
        """Clean up resources when object is destroyed."""
        self.logger.info("SemanticModelManager instance destroyed")
