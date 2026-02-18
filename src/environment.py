"""Multi-turn tool-use environment for RL training."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import random
import uuid


class ActionType(Enum):
    TEXT = "text"
    TOOL_CALL = "tool_call"
    FINISH = "finish"


@dataclass
class Action:
    """Agent action in the environment."""
    type: ActionType
    content: Optional[str] = None
    tool_name: Optional[str] = None
    tool_arguments: Optional[dict] = None
    answer: Optional[str] = None


@dataclass
class Episode:
    """Active episode state."""
    episode_id: str
    task_description: str
    conversation: list[dict] = field(default_factory=list)
    tool_outputs: list[dict] = field(default_factory=list)
    turn_count: int = 0
    max_turns: int = 10
    done: bool = False
    success: bool = False

    @classmethod
    def create(cls, task_description: str, max_turns: int = 10) -> "Episode":
        return cls(
            episode_id=str(uuid.uuid4()),
            task_description=task_description,
            max_turns=max_turns,
        )


# --- Mock tools (inline) ---

MOCK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code in a sandboxed environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                },
                "required": ["path"],
            },
        },
    },
]

SAMPLE_TASKS = [
    "Write a Python function to calculate fibonacci numbers and test it.",
    "Create a file with your favorite quotes and read it back.",
    "Calculate the sum of squares from 1 to 100.",
    "List the files in the workspace and describe what you see.",
]


def _execute_mock_tool(name: str, arguments: dict) -> tuple[str, bool]:
    """Execute a mock tool and return (result, is_valid)."""
    if name == "calculator":
        expr = arguments.get("expression", "")
        try:
            allowed = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
            result = eval(expr, {"__builtins__": {}}, allowed)
            return str(result), True
        except Exception as e:
            return f"Error: {e}", False

    if name == "execute_code":
        code = arguments.get("code", "")
        return f"[Mock] Code executed successfully\nOutput: executed {len(code)} chars", True

    if name == "read_file":
        path = arguments.get("path", "unknown")
        return f"[Mock] Contents of {path}:\nLine 1\nLine 2\nLine 3", True

    return f"Unknown tool: {name}", False


class Environment:
    """Concrete multi-turn tool-use environment with mock tools."""

    def __init__(
        self,
        max_turns: int = 10,
        turn_penalty: float = -0.1,
        success_reward: float = 1.0,
        failure_reward: float = -1.0,
    ):
        self.max_turns = max_turns
        self.turn_penalty = turn_penalty
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self._episode: Optional[Episode] = None

    def reset(self, task_description: Optional[str] = None) -> dict:
        """Reset environment. Returns observation dict."""
        if task_description is None:
            task_description = random.choice(SAMPLE_TASKS)

        self._episode = Episode.create(task_description, self.max_turns)
        self._episode.conversation.append({
            "role": "system",
            "content": f"You are a helpful assistant. Complete the following task:\n\n{task_description}",
        })
        return self._observation()

    def step(self, action: Action) -> tuple[dict, float, bool, dict]:
        """Execute one step. Returns (observation, reward, done, info)."""
        ep = self._episode
        if ep is None:
            raise RuntimeError("Call reset() first")
        if ep.done:
            raise RuntimeError("Episode done. Call reset().")

        reward = self.turn_penalty
        info = {"action_type": action.type.value}

        if action.type == ActionType.TEXT:
            ep.conversation.append({"role": "assistant", "content": action.content})

        elif action.type == ActionType.TOOL_CALL:
            call_id = f"call_{len(ep.tool_outputs)}"
            ep.conversation.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": action.tool_name, "arguments": action.tool_arguments},
                }],
            })
            result, valid = _execute_mock_tool(action.tool_name, action.tool_arguments or {})
            ep.tool_outputs.append({"tool": action.tool_name, "result": result, "valid": valid})
            ep.conversation.append({"role": "tool", "tool_call_id": call_id, "content": result})
            if not valid:
                reward += self.failure_reward
            info["tool_valid"] = valid

        elif action.type == ActionType.FINISH:
            success = bool(
                action.answer
                and any(w in action.answer.lower() for w in ["complete", "done", "finished"])
            )
            ep.done = True
            ep.success = success
            reward += self.success_reward if success else 0.0
            info["task_success"] = success

        ep.turn_count += 1
        truncated = ep.turn_count >= ep.max_turns and not ep.done
        if truncated:
            ep.done = True
            info["truncated"] = True

        return self._observation(), reward, ep.done, info

    def _observation(self) -> dict:
        ep = self._episode
        return {
            "conversation": ep.conversation.copy(),
            "available_tools": MOCK_TOOLS,
            "task_description": ep.task_description,
            "turn_count": ep.turn_count,
            "max_turns": ep.max_turns,
            "done": ep.done,
            "success": ep.success,
        }

    def execute_tool(self, name: str, arguments: dict) -> tuple[str, bool]:
        """Execute a tool by name. Returns (result_string, is_valid)."""
        return _execute_mock_tool(name, arguments)

    @property
    def episode(self) -> Optional[Episode]:
        return self._episode
