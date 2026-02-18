"""Dataset loading for multi-turn RL training.

Provides Task dataclass and loaders for GSM8K (math) and MBPP (code).
"""

from dataclasses import dataclass, field
from typing import Optional
import re


CALCULATOR_TOOL = {
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
}

EXECUTE_CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_code",
        "description": "Execute Python code and return the output.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
            },
            "required": ["code"],
        },
    },
}


@dataclass
class Task:
    """A single training task with verifiable reward."""
    id: str
    query: str
    tools: list[dict]
    ground_truth: Optional[str] = None
    test_code: Optional[str] = None
    reward_type: str = "exact_match"
    follow_ups: list[str] = field(default_factory=list)


def load_gsm8k(split: str = "train", n: int = 1000) -> list[Task]:
    """Load GSM8K math dataset. Requires `datasets` library."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    tasks = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        # Ground truth: number after "####" in answer
        answer_text = row["answer"]
        match = re.search(r"####\s*(.+)", answer_text)
        gt = match.group(1).strip().replace(",", "") if match else ""
        tasks.append(Task(
            id=f"gsm8k_{i}",
            query=row["question"],
            tools=[CALCULATOR_TOOL, EXECUTE_CODE_TOOL],
            ground_truth=gt,
            reward_type="gsm8k",
        ))
    return tasks


def load_mbpp(split: str = "train", n: int = 500) -> list[Task]:
    """Load MBPP code dataset. Requires `datasets` library."""
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
    tasks = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        tasks.append(Task(
            id=f"mbpp_{i}",
            query=row["text"],
            tools=[EXECUTE_CODE_TOOL],
            test_code="\n".join(row["test_list"]),
            reward_type="code_exec",
        ))
    return tasks


_LOADERS = {
    "gsm8k": load_gsm8k,
    "mbpp": load_mbpp,
}


def load_tasks(name: str, **kwargs) -> list[Task]:
    """Load tasks by dataset name."""
    loader = _LOADERS.get(name)
    if loader is None:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(_LOADERS.keys())}")
    return loader(**kwargs)
