"""Smoke tests for Multi-turn RL — no GPU or model downloads needed."""

import sys
from pathlib import Path

import pytest
import torch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


# --- Environment ---

def test_environment_reset():
    from src.environment import Environment
    env = Environment(max_turns=5)
    obs = env.reset("Test task")
    assert obs["task_description"] == "Test task"
    assert obs["turn_count"] == 0
    assert not obs["done"]
    assert len(obs["conversation"]) == 1  # system message


def test_environment_step_text():
    from src.environment import Environment, Action, ActionType
    env = Environment(max_turns=5)
    env.reset("Test task")
    obs, reward, done, info = env.step(Action(type=ActionType.TEXT, content="Hello"))
    assert info["action_type"] == "text"
    assert reward == env.turn_penalty
    assert not done


def test_environment_step_tool_call():
    from src.environment import Environment, Action, ActionType
    env = Environment(max_turns=5)
    env.reset("Calculate 2+2")
    obs, reward, done, info = env.step(
        Action(type=ActionType.TOOL_CALL, tool_name="calculator", tool_arguments={"expression": "2+2"})
    )
    assert info["tool_valid"] is True
    assert env.episode.tool_outputs[-1]["result"] == "4"


def test_environment_step_finish():
    from src.environment import Environment, Action, ActionType
    env = Environment(max_turns=5, success_reward=1.0)
    env.reset("Test")
    obs, reward, done, info = env.step(Action(type=ActionType.FINISH, answer="Task completed"))
    assert done
    assert info["task_success"] is True
    assert reward == env.turn_penalty + env.success_reward


def test_environment_truncation():
    from src.environment import Environment, Action, ActionType
    env = Environment(max_turns=2)
    env.reset("Test")
    env.step(Action(type=ActionType.TEXT, content="a"))
    obs, reward, done, info = env.step(Action(type=ActionType.TEXT, content="b"))
    assert done
    assert info.get("truncated") is True


def test_environment_execute_tool():
    from src.environment import Environment
    env = Environment()
    result, valid = env.execute_tool("calculator", {"expression": "3*7"})
    assert valid is True
    assert result == "21"


def test_environment_execute_tool_unknown():
    from src.environment import Environment
    env = Environment()
    result, valid = env.execute_tool("nonexistent", {})
    assert valid is False


# --- Policy (config only, no model load) ---

def test_policy_config():
    from src.policy import PolicyConfig
    cfg = PolicyConfig(model_name="test-model", lora_r=8)
    assert cfg.lora_r == 8
    assert cfg.use_lora is True


def test_policy_config_default_model():
    from src.policy import PolicyConfig
    cfg = PolicyConfig()
    assert cfg.model_name == "Qwen/Qwen3-1.7B"


def test_policy_init():
    from src.policy import QwenPolicy, PolicyConfig, HAS_TRANSFORMERS
    if not HAS_TRANSFORMERS:
        pytest.skip("transformers not installed")
    cfg = PolicyConfig(model_name="test-model")
    p = QwenPolicy(cfg)
    assert p.config.model_name == "test-model"
    assert p._model is None  # not loaded yet


# --- GRPO ---

def test_grpo_advantages():
    from src.grpo import compute_advantages
    rewards = torch.tensor([1.0, 2.0, 3.0, 0.0, 1.0, 2.0])
    group_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    adv = compute_advantages(rewards, group_ids, normalize=False)
    # Group 0 mean = 2.0, Group 1 mean = 1.0
    assert torch.allclose(adv[0], torch.tensor(-1.0))
    assert torch.allclose(adv[1], torch.tensor(0.0))
    assert torch.allclose(adv[2], torch.tensor(1.0))
    assert torch.allclose(adv[3], torch.tensor(-1.0))
    assert torch.allclose(adv[5], torch.tensor(1.0))


def test_grpo_advantages_normalized():
    from src.grpo import compute_advantages
    rewards = torch.tensor([1.0, 3.0, 5.0, 7.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    adv = compute_advantages(rewards, group_ids, normalize=True)
    # Each group of 2: normalized so std ~1
    assert abs(adv[0].item() + adv[1].item()) < 1e-5  # sum to ~0 within group


def test_grpo_batch_creation():
    from src.grpo import GRPOBatch
    B, IL, RL = 4, 10, 20
    batch = GRPOBatch(
        input_ids=torch.randint(0, 100, (B, IL)),
        attention_mask=torch.ones(B, IL, dtype=torch.long),
        response_ids=torch.randint(0, 100, (B, RL)),
        response_mask=torch.ones(B, RL, dtype=torch.long),
        rewards=torch.randn(B),
        group_ids=torch.tensor([0, 0, 1, 1]),
    )
    assert batch.input_ids.shape == (B, IL)
    assert batch.rewards.shape == (B,)


def test_grpo_batch_with_loss_mask():
    from src.grpo import GRPOBatch
    B, IL, RL = 4, 10, 20
    loss_mask = torch.ones(B, RL)
    loss_mask[:, 5:10] = 0  # mask out some tokens
    batch = GRPOBatch(
        input_ids=torch.randint(0, 100, (B, IL)),
        attention_mask=torch.ones(B, IL, dtype=torch.long),
        response_ids=torch.randint(0, 100, (B, RL)),
        response_mask=torch.ones(B, RL, dtype=torch.long),
        rewards=torch.randn(B),
        group_ids=torch.tensor([0, 0, 1, 1]),
        loss_mask=loss_mask,
    )
    assert batch.loss_mask is not None
    assert batch.loss_mask.shape == (B, RL)
    assert batch.loss_mask[0, 0] == 1.0
    assert batch.loss_mask[0, 5] == 0.0


# --- Rollout: parse_tool_call ---

def test_parse_tool_call_valid():
    from src.rollout import parse_tool_call
    text = '<think>Let me calculate</think>\n<tool_call>\n{"name": "calculator", "arguments": {"expression": "2+2"}}\n</tool_call>'
    result = parse_tool_call(text)
    assert result is not None
    assert result["name"] == "calculator"
    assert result["arguments"]["expression"] == "2+2"


def test_parse_tool_call_no_match():
    from src.rollout import parse_tool_call
    result = parse_tool_call("The answer is 42.")
    assert result is None


def test_parse_tool_call_invalid_json():
    from src.rollout import parse_tool_call
    result = parse_tool_call("<tool_call>not valid json</tool_call>")
    assert result is None


def test_parse_tool_call_execute_code():
    from src.rollout import parse_tool_call
    text = '<tool_call>\n{"name": "execute_code", "arguments": {"code": "print(42)"}}\n</tool_call>'
    result = parse_tool_call(text)
    assert result is not None
    assert result["name"] == "execute_code"
    assert result["arguments"]["code"] == "print(42)"


# --- Data ---

def test_task_dataclass():
    from src.data import Task
    task = Task(
        id="test_0",
        query="What is 2+2?",
        tools=[],
        ground_truth="4",
        reward_type="gsm8k",
    )
    assert task.id == "test_0"
    assert task.follow_ups == []
    assert task.test_code is None


def test_task_with_follow_ups():
    from src.data import Task
    task = Task(
        id="test_1",
        query="First question",
        tools=[],
        follow_ups=["Second question", "Third question"],
    )
    assert len(task.follow_ups) == 2


# --- Rewards ---

def test_gsm8k_reward_correct():
    from src.rewards import compute_reward
    from src.data import Task
    task = Task(id="t", query="q", tools=[], ground_truth="42", reward_type="gsm8k")
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is the answer?"},
        {"role": "assistant", "content": "<think>The answer is 42.</think>\nThe answer is 42."},
    ]
    assert compute_reward(messages, task) == 1.0


def test_gsm8k_reward_incorrect():
    from src.rewards import compute_reward
    from src.data import Task
    task = Task(id="t", query="q", tools=[], ground_truth="42", reward_type="gsm8k")
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is the answer?"},
        {"role": "assistant", "content": "The answer is 99."},
    ]
    assert compute_reward(messages, task) == 0.0


def test_gsm8k_reward_no_number():
    from src.rewards import compute_reward
    from src.data import Task
    task = Task(id="t", query="q", tools=[], ground_truth="42", reward_type="gsm8k")
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is the answer?"},
        {"role": "assistant", "content": "I don't know."},
    ]
    assert compute_reward(messages, task) == 0.0


def test_exact_match_reward():
    from src.rewards import compute_reward
    from src.data import Task
    task = Task(id="t", query="q", tools=[], ground_truth="hello world", reward_type="exact_match")
    messages = [
        {"role": "assistant", "content": "The answer is hello world here."},
    ]
    assert compute_reward(messages, task) == 1.0


# --- Build loss mask (with mock tokenizer) ---

class MockTokenizer:
    """Minimal tokenizer mock for testing build_loss_mask."""
    def __init__(self):
        # Simulate token IDs
        self._vocab = {
            "<|im_start|>": 100,
            "<|im_end|>": 101,
            "system": 200,
            "user": 201,
            "assistant": 202,
            "tool": 203,
            "\n": 204,
            "hello": 205,
            "world": 206,
            "result": 207,
        }
        self._id_to_tok = {v: k for k, v in self._vocab.items()}
        self.unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        # Simple: return [vocab[text]] if text is a single token
        if text in self._vocab:
            return [self._vocab[text]]
        return [self.unk_token_id]

    def decode(self, token_ids):
        if isinstance(token_ids, list):
            return "".join(self._id_to_tok.get(t, "?") for t in token_ids)
        return self._id_to_tok.get(token_ids, "?")


def test_build_loss_mask():
    from src.rollout import build_loss_mask
    tok = MockTokenizer()

    # Sequence: <|im_start|>system\nhello<|im_end|><|im_start|>user\nworld<|im_end|><|im_start|>assistant\nhello world<|im_end|>
    token_ids = [
        100, 200, 204, 205, 101,           # system block: im_start, system, \n, hello, im_end
        100, 201, 204, 206, 101,           # user block: im_start, user, \n, world, im_end
        100, 202, 204, 205, 206, 101,     # assistant block: im_start, assistant, \n, hello, world, im_end
    ]

    mask = build_loss_mask(token_ids, tok)
    assert mask.shape[0] == len(token_ids)

    # System tokens (indices 0-4) should be 0
    assert mask[0].item() == 0.0  # <|im_start|>
    assert mask[3].item() == 0.0  # hello (system content)
    assert mask[4].item() == 0.0  # <|im_end|>

    # User tokens (indices 5-9) should be 0
    assert mask[8].item() == 0.0  # world (user content)

    # Assistant tokens: content and im_end should be 1
    assert mask[13].item() == 1.0  # hello (assistant content)
    assert mask[14].item() == 1.0  # world (assistant content)
    assert mask[15].item() == 1.0  # <|im_end|> (assistant)


def test_build_loss_mask_with_tool():
    from src.rollout import build_loss_mask
    tok = MockTokenizer()

    # assistant -> tool -> assistant
    token_ids = [
        100, 202, 204, 205, 101,           # assistant: hello
        100, 203, 204, 207, 101,           # tool: result
        100, 202, 204, 206, 101,           # assistant: world
    ]

    mask = build_loss_mask(token_ids, tok)

    # First assistant content
    assert mask[3].item() == 1.0   # hello
    assert mask[4].item() == 1.0   # im_end (assistant)

    # Tool content
    assert mask[8].item() == 0.0   # result (tool)

    # Second assistant content
    assert mask[13].item() == 1.0  # world
    assert mask[14].item() == 1.0  # im_end (assistant)
