from abc import ABC, abstractmethod
from typing import Any

from cafe.utils.logger import setup_logger


class BaseAgent(ABC):
    """Abstract base class for agents."""

    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)

    def __call__(self, *args, **kwargs):
        """Make the agent callable."""
        return self.run(*args, **kwargs)

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Execute the agent's main functionality."""
        pass
