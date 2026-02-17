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


# --- Policy (config only, no model load) ---

def test_policy_config():
    from src.policy import PolicyConfig
    cfg = PolicyConfig(model_name="test-model", lora_r=8)
    assert cfg.lora_r == 8
    assert cfg.use_lora is True


def test_policy_init():
    from src.policy import QwenPolicy, PolicyConfig
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


# --- Rollout (mock, no model) ---

def test_rollout_parse_action():
    from src.rollout import _parse_action
    from src.environment import ActionType

    a1 = _parse_action("I have finished the task")
    assert a1.type == ActionType.FINISH

    a2 = _parse_action("Let me think about this")
    assert a2.type == ActionType.TEXT
