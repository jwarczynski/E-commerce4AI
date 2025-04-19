import os
import time
from pathlib import Path
from typing import Any, Dict

import yaml

from cafe.models.semantic_model_graph import SemanticModelGraph
from cafe.utils.logger import setup_logger


class SemanticModelManager:
    """Manages semantic models, including loading, updating, and graph tracking."""

    def __init__(self):
        self.logger = setup_logger(__name__)
        self.graph = SemanticModelGraph()

    def load_yaml(self, file_path: str | Path) -> str:
        """Load a YAML file as a string."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Semantic model file not found: {file_path}")
        with open(file_path, 'r') as file:
            return file.read()

    def parse_yaml(self, yaml_content: str) -> Dict[str, Any]:
        """Parse YAML content into a dictionary."""
        return yaml.safe_load(yaml_content)

    def update_verified_queries(self, file_path: str | Path, query_name: str, question: str, sql: str, verified_by: str = "system") -> None:
        """Update the verified_queries section of a semantic model."""
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
        self.logger.info(f"Updated verified_queries in {file_path}")

    def create_new_semantic_model(
            self,
            original_model_path: str | Path,
            new_model_path: str,
            new_table: Dict[str, Any]
    ) -> None:
        """Create a new semantic model with an extended table, excluding verified queries."""
        original_model = self.parse_yaml(self.load_yaml(original_model_path))
        new_model = original_model.copy()
        new_model["name"] = f"{original_model['name']}_extended"
        new_model["tables"].append(new_table)
        new_model.pop("verified_queries", None)  # Exclude verified queries
        with open(new_model_path, 'w') as file:
            yaml.safe_dump(new_model, file)
        self.graph.add_edge(original_model_path, new_model_path)
        self.logger.info(f"Created new semantic model: {new_model_path}")
