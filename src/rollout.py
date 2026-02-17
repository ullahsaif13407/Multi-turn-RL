"""Rollout collection for multi-turn RL.

Collects episodes by running the policy in the environment,
then packages results into GRPOBatch for training.
"""

from typing import Any, Optional
import logging
import uuid

from src.environment import Action, ActionType, Environment
from src.grpo import GRPOBatch

import torch

logger = logging.getLogger(__name__)


def collect_episode(
    policy: Any,
    env: Environment,
    max_turns: int = 10,
    max_tokens: int = 512,
) -> dict:
    """Collect one episode. Returns dict with turns, total_reward, success."""
    obs = env.reset()
    turns = []
    total_reward = 0.0

    for t in range(max_turns):
        # Generate action from conversation
        messages = obs["conversation"]
        try:
            output_ids = policy.generate(messages=messages, max_new_tokens=max_tokens)
            text = policy.tokenizer.decode(output_ids[0][len(policy.tokenizer.encode(
                policy.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )):], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            text = f"Error: {e}"

        # Parse into action
        action = _parse_action(text)

        # Step
        obs, reward, done, info = env.step(action)
        turns.append({"text": text, "action_type": action.type.value, "reward": reward})
        total_reward += reward

        if done:
            break

    return {
        "task": env.episode.task_description,
        "turns": turns,
        "total_reward": total_reward,
        "success": env.episode.success,
        "length": len(turns),
    }


def _parse_action(text: str) -> Action:
    """Simple heuristic to convert generated text into an Action."""
    lower = text.lower().strip()
    if any(w in lower for w in ["task complete", "task completed", "finished", "done"]):
        return Action(type=ActionType.FINISH, answer=text)
    return Action(type=ActionType.TEXT, content=text)


def collect_grpo_batch(
    policy: Any,
    env: Environment,
    tokenizer: Any,
    num_prompts: int = 4,
    group_size: int = 4,
    max_turns: int = 10,
    device: str = "cuda",
) -> tuple[GRPOBatch, dict]:
    """Collect grouped episodes and build a GRPOBatch.

    Returns (batch, stats_dict).
    """
    all_input_ids, all_attn, all_resp_ids, all_resp_mask = [], [], [], []
    all_rewards, all_group_ids = [], []
    all_episodes = []

    for prompt_idx in range(num_prompts):
        for _ in range(group_size):
            ep = collect_episode(policy, env, max_turns)
            all_episodes.append(ep)

            # Encode prompt
            prompt_enc = tokenizer(ep["task"], return_tensors="pt", truncation=True, max_length=512)
            all_input_ids.append(prompt_enc["input_ids"][0])
            all_attn.append(prompt_enc["attention_mask"][0])

            # Encode response (concatenate turn texts)
            response_text = " ".join(t["text"] for t in ep["turns"])
            resp_enc = tokenizer(response_text, return_tensors="pt", truncation=True, max_length=1024)
            all_resp_ids.append(resp_enc["input_ids"][0])
            all_resp_mask.append(resp_enc["attention_mask"][0])

            all_rewards.append(ep["total_reward"])
            all_group_ids.append(prompt_idx)

    # Pad and stack
    input_ids = _pad(all_input_ids, tokenizer.pad_token_id or 0).to(device)
    attn_mask = _pad(all_attn, 0).to(device)
    resp_ids = _pad(all_resp_ids, tokenizer.pad_token_id or 0).to(device)
    resp_mask = _pad(all_resp_mask, 0).to(device)

    batch = GRPOBatch(
        input_ids=input_ids,
        attention_mask=attn_mask,
        response_ids=resp_ids,
        response_mask=resp_mask,
        rewards=torch.tensor(all_rewards, device=device),
        group_ids=torch.tensor(all_group_ids, device=device),
    )

    # Stats
    rewards = [ep["total_reward"] for ep in all_episodes]
    stats = {
        "num_episodes": len(all_episodes),
        "reward_mean": sum(rewards) / len(rewards),
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "success_rate": sum(ep["success"] for ep in all_episodes) / len(all_episodes),
        "avg_length": sum(ep["length"] for ep in all_episodes) / len(all_episodes),
    }
    return batch, stats


def _pad(tensors: list[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    """Pad list of 1D tensors to same length."""
    max_len = max(t.size(0) for t in tensors)
    out = torch.full((len(tensors), max_len), pad_value, dtype=torch.long)
    for i, t in enumerate(tensors):
        out[i, :t.size(0)] = t
    return out
