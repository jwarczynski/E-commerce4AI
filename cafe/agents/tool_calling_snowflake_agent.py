import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml
from rich.panel import Panel
from rich.text import Text
from smolagents import (
    ActionStep,
    AgentParsingError,
    AgentToolCallError, AgentToolExecutionError, ChatMessage,
    LogLevel,
    MessageRole,
    MultiStepAgent,
    PromptTemplates, Tool,
    YELLOW_HEX, populate_template,
)
from smolagents.memory import Message, TaskStep, ToolCall
from smolagents.models import ChatMessageToolCall, ChatMessageToolCallDefinition

from cafe.core.snowflake_client import SnowflakeClient


@dataclass
class ToolInputSchema:
    type: str
    properties: Dict[str, Any]
    required: List[str]


@dataclass
class ToolSpec:
    type: str
    name: str
    input_schema: ToolInputSchema


@dataclass
class TaskActionStep(ActionStep):
    content_list: dict[str, Any] = None


class SnowFlakeMessage(Message):
    content_list: dict[str, Any]


class SnowflakeApiModel:
    def __init__(self, snoflake_client: SnowflakeClient):
        self.snowflake_client = snoflake_client

    def __call__(self, messages: List[Message], tools_to_call_from: List[Tool]=None, **kwargs) -> ChatMessage:
        """Call the Snowflake API with the provided arguments."""
        data = {
            "model": "claude-3-5-sonnet",
            "messages": messages,
            "tools": self._parse_tools_to_call_from(tools_to_call_from) if tools_to_call_from else [],
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False
        }

        response = self.snowflake_client.call_cortex_llm(data)
        raw_answer = response["choices"][0]["message"]

        return ChatMessage(
            role="assistant",
            content=self._extract_answer_text(response),
            tool_calls=self._extract_tool_use(response),
            raw=raw_answer
        )

    def _parse_tools_to_call_from(self, tools_to_call_from: List[Tool]) -> List[Dict]:
        """Parse tools to the format expected by the Snowflake API."""
        tools = []
        for tool in tools_to_call_from:
            try:
                # Create properties dictionary, modifying 'final_answer' tool if needed
                properties = tool.inputs
                if tool.name == "final_answer":
                    if "answer" not in tool.inputs:
                        raise ValueError("final_answer tool must have an 'answer' property")
                    # Ensure 'answer' property uses type 'string'
                    properties = {
                        key: (
                            {
                                "description": prop.get("description", "The final answer to the problem"),
                                "type": "string"
                            }
                            if key == "answer" and isinstance(prop, dict)
                            else prop
                        )
                        for key, prop in tool.inputs.items()
                    }

                tool_spec = ToolSpec(
                    type="generic",
                    name=tool.name,
                    input_schema=ToolInputSchema(
                        type="object",
                        properties=properties,
                        required=list(tool.inputs.keys()) if tool.inputs else []
                    )
                )
                # Convert ToolSpec and its nested ToolInputSchema to a dictionary
                tool_spec_dict = asdict(tool_spec)
                tools.append({"tool_spec": tool_spec_dict})
            except Exception as e:
                raise ValueError(f"Failed to parse tool '{tool.name}': {str(e)}")
        return tools

    def _extract_answer_text(self, response: Dict) -> str:
        """Extract the answer text from the API response."""
        raw_answer = response["choices"][0]["message"]
        return raw_answer.get("content", "")

    def _extract_tool_use(self, response: Dict) -> Optional[List[ChatMessageToolCall]]:
        """Extract tool use data from the API response."""
        raw_answer = response["choices"][0]["message"]
        content_list = raw_answer.get("content_list", [])

        if not content_list:
            return None

        tool_calls = []
        for item in content_list:
            if item.get("type") == "tool_use":
                tool_use = item.get("tool_use", {})
                tool_call_def = ChatMessageToolCallDefinition(
                    arguments=tool_use.get("input", {}),
                    name=tool_use.get("name", ""),
                    description=None
                )
                tool_call = ChatMessageToolCall(
                    function=tool_call_def,
                    id=tool_use.get("tool_use_id", ""),
                    type="tool_use"
                )
                tool_calls.append(tool_call)

        return tool_calls if tool_calls else None


class ToolCallingSnowflakeAgent(MultiStepAgent):

    def __init__(
            self,
            tools: List[Tool],
            model: Callable[[List[Dict[str, str]]], ChatMessage],
            prompt_templates: Optional[PromptTemplates] = None,
            planning_interval: Optional[int] = None,
            prompt_templates_path: Optional[Path] = None,
            **kwargs,
    ):
        prompt_templates_path = prompt_templates_path or (Path(__file__).parent.parent / "prompts" / "tool_calling_snowflake_agent.yaml")
        prompt_templates = prompt_templates or yaml.safe_load(
            prompt_templates_path.read_text()
        )
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            **kwargs,
        )

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={"tools": self.tools, "managed_agents": self.managed_agents},
        )
        return system_prompt

    def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """
                Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
                Returns None if the step is not final.
                """
        memory_messages = self.write_memory_to_messages()

        self.input_messages = memory_messages

        # Add new step in logs
        memory_step.model_input_messages = memory_messages.copy()

        try:
            model_message: ChatMessage = self.model(
                memory_messages,
                tools_to_call_from=list(self.tools.values()),
                stop_sequences=["Observation:", "Calling tools:"],
            )
            memory_step.model_output_message = model_message.content
        except Exception as e:
            raise AgentParsingError(f"Error while generating or parsing output:\n{e}", self.logger) from e

        self.logger.log_markdown(
            content=model_message.content if model_message.content else str(model_message.raw),
            title="Output message of the LLM:",
            level=LogLevel.DEBUG,
        )

        if model_message.tool_calls is None or len(model_message.tool_calls) == 0:
            raise AgentParsingError(
                "Model did not call any tools. Call `final_answer` tool to return a final answer.", self.logger
            )

        tool_call = model_message.tool_calls[0]
        tool_name, tool_call_id = tool_call.function.name, tool_call.id
        tool_arguments = tool_call.function.arguments
        memory_step.model_output = [item for item in model_message.raw.get("content_list", []) if item.get("type") == "tool_use"]
        memory_step.tool_calls = [ToolCall(name=tool_name, arguments=tool_arguments, id=tool_call_id)]

        # Execute
        self.logger.log(
            Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
            level=LogLevel.INFO,
        )
        if tool_name == "final_answer":
            if isinstance(tool_arguments, dict):
                if "answer" in tool_arguments:
                    answer = tool_arguments["answer"]
                else:
                    answer = tool_arguments
            else:
                answer = tool_arguments
            if (
                    isinstance(answer, str) and answer in self.state.keys()
            ):  # if the answer is a state variable, return the value
                final_answer = self.state[answer]
                self.logger.log(
                    f"[bold {YELLOW_HEX}]Final answer:[/bold {YELLOW_HEX}] Extracting key '{answer}' from state to return value '{final_answer}'.",
                    level=LogLevel.INFO,
                )
            else:
                final_answer = answer
                self.logger.log(
                    Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                    level=LogLevel.INFO,
                )

            memory_step.action_output = final_answer
            return final_answer
        else:
            if tool_arguments is None:
                tool_arguments = {}
            observation = self.execute_tool_call(tool_name, tool_arguments, tool_call_id)
            updated_information = str(observation).strip()
            self.logger.log(
                f"Observations: {updated_information.replace('[', '|')}",  # escape potential rich-tag-like components
                level=LogLevel.INFO,
            )
            memory_step.observations = observation
            return None

    def write_memory_to_messages(
            self,
            summary_mode: Optional[bool] = False,
    ) -> List[Dict[str, str]]:
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """
        messages = system_prompt_to_messages(self.memory.system_prompt.system_prompt)
        for memory_step in self.memory.steps:
            if isinstance(memory_step, (ActionStep, TaskActionStep)):
                messages.extend(action_step_to_messages(memory_step, summary_mode=summary_mode))
            elif isinstance(memory_step, TaskStep):
                messages.append(
                    Message(role=MessageRole.USER.value, content=f"New task:\n{memory_step.task}")
                )

        return messages

    def execute_tool_call(self, tool_name: str, arguments: Union[Dict[str, str], str], tool_id: str) -> Any:
        """
        Execute a tool or managed agent with the provided arguments.

        The arguments are replaced with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the tool or managed agent to execute.
            arguments (dict[str, str] | str): Arguments passed to the tool call.
        """
        # Check if the tool exists
        available_tools = {**self.tools, **self.managed_agents}
        if tool_name not in available_tools:
            raise AgentToolExecutionError(
                f"Unknown tool {tool_name}, should be one of: {', '.join(available_tools)}.", self.logger
            )

        # Get the tool and substitute state variables in arguments
        tool = available_tools[tool_name]
        arguments = self._substitute_state_variables(arguments)
        is_managed_agent = tool_name in self.managed_agents

        try:
            # Call tool with appropriate arguments
            if isinstance(arguments, dict):
                tool_res = tool(**arguments) if is_managed_agent else tool(**arguments, sanitize_inputs_outputs=True)
            elif isinstance(arguments, str):
                tool_res = tool(arguments) if is_managed_agent else tool(arguments, sanitize_inputs_outputs=True)
            else:
                raise TypeError(f"Unsupported arguments type: {type(arguments)}")

            return [
                {
                    "type": "tool_results",
                    "tool_results": {
                        "tool_use_id": tool_id,
                        "name": tool_name,
                        "content": [
                            {
                                "type": "text",
                                "text": tool_res
                            }
                        ]
                    }
                }
            ]

        except TypeError as e:
            # Handle invalid arguments
            description = getattr(tool, "description", "No description")
            if is_managed_agent:
                error_msg = (
                    f"Invalid request to team member '{tool_name}' with arguments {json.dumps(arguments)}: {e}\n"
                    "You should call this team member with a valid request.\n"
                    f"Team member description: {description}"
                )
            else:
                error_msg = (
                    f"Invalid call to tool '{tool_name}' with arguments {json.dumps(arguments)}: {e}\n"
                    "You should call this tool with correct input arguments.\n"
                    f"Expected inputs: {json.dumps(tool.inputs)}\n"
                    f"Returns output type: {tool.output_type}\n"
                    f"Tool description: '{description}'"
                )
            raise AgentToolCallError(error_msg, self.logger) from e

        except Exception as e:
            # Handle execution errors
            if is_managed_agent:
                error_msg = (
                    f"Error executing request to team member '{tool_name}' with arguments {json.dumps(arguments)}: {e}\n"
                    "Please try again or request to another team member"
                )
            else:
                error_msg = (
                    f"Error executing tool '{tool_name}' with arguments {json.dumps(arguments)}: {type(e).__name__}: {e}\n"
                    "Please try again or use another tool"
                )
            raise AgentToolExecutionError(error_msg, self.logger) from e

    def _substitute_state_variables(self, arguments: Union[Dict[str, str], str]) -> Union[Dict[str, Any], str]:
        """Replace string values in arguments with their corresponding state values if they exist."""
        if isinstance(arguments, dict):
            return {
                key: self.state.get(value, value) if isinstance(value, str) else value
                for key, value in arguments.items()
            }
        return arguments

    def _create_action_step(self, step_start_time: float, images: List["PIL.Image.Image"] | None) -> ActionStep:
        return TaskActionStep(step_number=self.step_number, start_time=step_start_time, observations_images=images)


def system_prompt_to_messages(system_prompt: str) -> List[Message]:
    """Converts the system prompt to a list of messages."""
    return [Message(role=MessageRole.SYSTEM, content=system_prompt)]


def action_step_to_messages(memory_step, summary_mode: bool = False, show_model_input_messages: bool = False) -> List[
    Message]:
    messages = []
    if memory_step.model_input_messages is not None and show_model_input_messages:
        messages.append(SnowFlakeMessage(role=MessageRole.SYSTEM, content=memory_step.model_input_messages))
    if memory_step.model_output is not None:
        messages.append(
            SnowFlakeMessage(role=MessageRole.ASSISTANT.value, content_list=memory_step.model_output, content=memory_step.model_output_message)
        )

    if memory_step.observations is not None:
        messages.append(
            SnowFlakeMessage(
                role=MessageRole.USER.value,
                content=memory_step.model_input_messages[-1]["content"],
                content_list=memory_step.observations,
            )
        )
    if memory_step.error is not None:
        error_message = (
                "Error:\n"
                + str(memory_step.error)
                + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
        )
        message_content = f"Call id: {memory_step.tool_calls[0].id}\n" if memory_step.tool_calls else ""
        message_content += error_message
        conten_list = [
          {
            "type": "tool_results",
            "tool_results": {
              "tool_use_id": memory_step.tool_calls[0].id,
              "name": memory_step.tool_calls[0].name,
              "content": [
                {
                  "type": "text",
                  "text": error_message
                }
              ]
            }
          }
        ] if memory_step.tool_calls else []
        messages.append(
            SnowFlakeMessage(role=MessageRole.USER, content=message_content, content_list=conten_list)
        )

    return messages
