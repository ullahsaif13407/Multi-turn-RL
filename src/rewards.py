"""Verifiable reward functions for multi-turn RL."""

import re
import subprocess
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data import Task


def compute_reward(messages: list[dict], task: "Task") -> float:
    """Dispatch to reward function based on task.reward_type."""
    fn = _REWARD_FNS.get(task.reward_type)
    if fn is None:
        raise ValueError(f"Unknown reward_type: {task.reward_type}")
    return fn(messages, task)


def _gsm8k_reward(messages: list[dict], task: "Task") -> float:
    """Compare final numeric answer to ground truth."""
    # Get last assistant message, strip <think> blocks
    last_assistant = ""
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            last_assistant = msg.get("content", "") or ""
            break
    text = re.sub(r"<think>.*?</think>", "", last_assistant, flags=re.DOTALL).strip()

    # Extract last number from assistant text
    numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
    if not numbers:
        return 0.0
    predicted = numbers[-1].replace(",", "")
    try:
        return 1.0 if float(predicted) == float(task.ground_truth) else 0.0
    except (ValueError, TypeError):
        return 0.0


def _code_exec_reward(messages: list[dict], task: "Task") -> float:
    """Run model's code + test assertions in a subprocess."""
    # Collect code from execute_code tool calls in the conversation
    code_blocks = []
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg.get("content", "") or ""
            # Extract code from tool_call blocks
            for match in re.finditer(
                r'<tool_call>\s*\{[^}]*"name"\s*:\s*"execute_code"[^}]*"code"\s*:\s*"(.*?)"[^}]*\}\s*</tool_call>',
                content, re.DOTALL,
            ):
                code_blocks.append(match.group(1))

    if not code_blocks and task.test_code:
        # Fallback: extract code from the final assistant message (non-tool response)
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                content = msg.get("content", "") or ""
                clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                # Look for code fenced blocks
                fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", clean, re.DOTALL)
                code_blocks.extend(fenced)
                break

    if not code_blocks:
        return 0.0

    full_code = "\n".join(code_blocks) + "\n" + (task.test_code or "")
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
            f.write(full_code)
            f.flush()
            result = subprocess.run(
                ["python", f.name],
                capture_output=True, text=True, timeout=10,
            )
        return 1.0 if result.returncode == 0 else 0.0
    except (subprocess.TimeoutExpired, Exception):
        return 0.0


def _exact_match_reward(messages: list[dict], task: "Task") -> float:
    """Simple exact match on last assistant message."""
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            content = msg.get("content", "") or ""
            text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if task.ground_truth and task.ground_truth.strip() in text:
                return 1.0
            return 0.0
    return 0.0


_REWARD_FNS = {
    "gsm8k": _gsm8k_reward,
    "code_exec": _code_exec_reward,
    "exact_match": _exact_match_reward,
}
