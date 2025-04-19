import networkx as nx
from cafe.utils.logger import setup_logger

class SemanticModelGraph:
    """Directed graph to track relationships between semantic models."""
    def __init__(self):
        self.graph = nx.DiGraph()
        self.logger = setup_logger(__name__)

    def add_edge(self, source: str, target: str) -> None:
        """Add an edge from source to target semantic model."""
        self.graph.add_edge(source, target)
        self.logger.info(f"Added edge: {source} -> {target}")

    def get_graph(self) -> nx.DiGraph:
        """Return the directed graph."""
        return self.graph