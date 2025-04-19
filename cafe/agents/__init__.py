from typing import Type, TypeVar

from cafe.agents.base_agent import BaseAgent
from cafe.agents.evaluating import EvaluatingAgent
from cafe.agents.feature_engineering import FeatureEngineeringAgent
from cafe.agents.judge import JudgeAgent

T = TypeVar("T", bound=BaseAgent)


class AgentFactory:
    """Factory for creating agent instances."""

    @staticmethod
    def create_agent(agent_type: str, *args, **kwargs) -> T:
        """Create an agent based on the specified type."""
        agents: dict[str, Type[BaseAgent]] = {
            "feature_engineering": FeatureEngineeringAgent,
            "judge": JudgeAgent,
            "evaluating": EvaluatingAgent
        }
        if agent_type not in agents:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return agents[agent_type](*args, **kwargs)
