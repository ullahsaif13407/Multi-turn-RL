"""Rollout collection for multi-turn RL with tool calling.

Collects episodes by running the policy in the environment,
then packages results into GRPOBatch for training.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import json
import logging
import re

import torch

from src.data import Task
from src.environment import Environment
from src.grpo import GRPOBatch
from src.rewards import compute_reward

logger = logging.getLogger(__name__)

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

SYSTEM_PROMPT = "You are a helpful assistant. Think step by step, and use tools when needed."


@dataclass
class EpisodeResult:
    """Result of a single rollout episode."""
    messages: list[dict]
    reward: float
    num_tool_calls: int = 0
    num_turns: int = 0


def parse_tool_call(text: str) -> Optional[dict]:
    """Extract tool call from assistant text. Returns {"name": ..., "arguments": ...} or None."""
    match = TOOL_CALL_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return None


def collect_episode(
    policy: Any,
    task: Task,
    env: Environment,
    max_tool_rounds: int = 5,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> EpisodeResult:
    """Collect one episode: generate → tool call → tool result → ... → final response."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.query},
    ]
    total_tool_calls = 0

    # Process initial query + any follow-ups
    all_queries = [task.query] + task.follow_ups
    for turn_idx, query in enumerate(all_queries):
        if turn_idx > 0:
            messages.append({"role": "user", "content": query})

        # Inner loop: generate, check for tool calls, execute, repeat
        for round_idx in range(max_tool_rounds):
            try:
                text = policy.generate_turn(
                    messages, tools=task.tools,
                    max_new_tokens=max_new_tokens, temperature=temperature,
                )
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                text = f"I encountered an error: {e}"

            # Strip trailing <|im_end|> if present (we add it via chat template role boundaries)
            text = text.rstrip()
            if text.endswith("<|im_end|>"):
                text = text[:-len("<|im_end|>")].rstrip()

            messages.append({"role": "assistant", "content": text})

            tool_call = parse_tool_call(text)
            if tool_call:
                name = tool_call.get("name", "")
                args = tool_call.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        args = {}
                result, _ = env.execute_tool(name, args)
                messages.append({"role": "tool", "content": json.dumps({"output": result})})
                total_tool_calls += 1
            else:
                break  # Final response, no more tool calls this turn

    reward = compute_reward(messages, task)
    return EpisodeResult(
        messages=messages,
        reward=reward,
        num_tool_calls=total_tool_calls,
        num_turns=len(all_queries),
    )


def build_loss_mask(token_ids: list[int], tokenizer: Any) -> torch.Tensor:
    """Build loss mask: 1 for assistant tokens, 0 for system/user/tool tokens.

    State machine scans for <|im_start|> followed by role tokens,
    then sets mask based on role.
    """
    n = len(token_ids)
    mask = torch.zeros(n, dtype=torch.float32)

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Token IDs for role identifiers
    # Qwen uses single tokens for role names after <|im_start|>
    assistant_token_ids = set(tokenizer.encode("assistant", add_special_tokens=False))

    i = 0
    while i < n:
        if token_ids[i] == im_start_id:
            # Look at next token(s) to determine role
            role_start = i + 1
            is_assistant = False
            if role_start < n and token_ids[role_start] in assistant_token_ids:
                is_assistant = True

            # Skip past <|im_start|>role\n to content
            i = role_start
            # Skip role token(s) and newline
            while i < n and token_ids[i] != im_end_id and token_ids[i] != im_start_id:
                # Check if we've passed the role header (first newline after im_start)
                decoded = tokenizer.decode([token_ids[i]])
                if "\n" in decoded:
                    i += 1
                    break
                i += 1
            else:
                # Hit im_end or im_start without finding newline — skip
                if i < n and token_ids[i] == im_end_id:
                    i += 1
                continue

            # Now i points to content tokens; set mask until <|im_end|>
            while i < n and token_ids[i] != im_end_id and token_ids[i] != im_start_id:
                if is_assistant:
                    mask[i] = 1.0
                i += 1
            # Also mask the <|im_end|> for assistant (it's part of the generation target)
            if i < n and token_ids[i] == im_end_id and is_assistant:
                mask[i] = 1.0
            if i < n:
                i += 1
        else:
            i += 1

    return mask


def collect_grpo_batch(
    policy: Any,
    tasks: list[Task],
    env: Environment,
    tokenizer: Any,
    group_size: int = 4,
    max_tool_rounds: int = 5,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    device: str = "cuda",
) -> tuple[GRPOBatch, dict]:
    """Collect grouped episodes and build a GRPOBatch with loss masks.

    For each task, collects group_size episodes. Tokenizes full conversations,
    splits into input (system + first user) and response (everything after),
    builds loss masks over response tokens.

    Returns (batch, stats_dict).
    """
    all_input_ids, all_attn = [], []
    all_resp_ids, all_resp_mask, all_loss_mask = [], [], []
    all_rewards, all_group_ids = [], []
    all_episodes = []

    for task_idx, task in enumerate(tasks):
        for _ in range(group_size):
            ep = collect_episode(
                policy, task, env,
                max_tool_rounds=max_tool_rounds,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            all_episodes.append(ep)

            # Tokenize full conversation
            template_kwargs = dict(
                tokenize=False, add_generation_prompt=False, enable_thinking=True,
            )
            if task.tools:
                template_kwargs["tools"] = task.tools
            full_text = tokenizer.apply_chat_template(ep.messages, **template_kwargs)
            full_enc = tokenizer(full_text, return_tensors="pt", truncation=True,
                                 max_length=tokenizer.model_max_length or 4096)
            full_ids = full_enc["input_ids"][0]

            # Tokenize prompt (system + first user) to find split point
            prompt_msgs = ep.messages[:2]  # system + first user
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
                **({"tools": task.tools} if task.tools else {}),
            )
            prompt_enc = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                                   max_length=tokenizer.model_max_length or 4096)
            prompt_len = prompt_enc["input_ids"].shape[1]

            # Split
            input_ids = full_ids[:prompt_len]
            resp_ids = full_ids[prompt_len:]

            if resp_ids.shape[0] == 0:
                # Degenerate case: no response tokens
                resp_ids = torch.tensor([tokenizer.eos_token_id], dtype=torch.long)

            all_input_ids.append(input_ids)
            all_attn.append(torch.ones_like(input_ids))
            all_resp_ids.append(resp_ids)
            all_resp_mask.append(torch.ones_like(resp_ids))

            # Build loss mask over response tokens
            lm = build_loss_mask(full_ids.tolist(), tokenizer)
            resp_loss_mask = lm[prompt_len:]
            if resp_loss_mask.shape[0] == 0:
                resp_loss_mask = torch.ones(resp_ids.shape[0], dtype=torch.float32)
            all_loss_mask.append(resp_loss_mask)

            all_rewards.append(ep.reward)
            all_group_ids.append(task_idx)

    # Pad and stack
    input_ids = _pad(all_input_ids, tokenizer.pad_token_id or 0).to(device)
    attn_mask = _pad(all_attn, 0).to(device)
    resp_ids = _pad(all_resp_ids, tokenizer.pad_token_id or 0).to(device)
    resp_mask = _pad(all_resp_mask, 0).to(device)
    loss_mask = _pad_float(all_loss_mask, 0.0).to(device)

    batch = GRPOBatch(
        input_ids=input_ids,
        attention_mask=attn_mask,
        response_ids=resp_ids,
        response_mask=resp_mask,
        rewards=torch.tensor(all_rewards, dtype=torch.float32, device=device),
        group_ids=torch.tensor(all_group_ids, dtype=torch.long, device=device),
        loss_mask=loss_mask,
    )

    rewards = [ep.reward for ep in all_episodes]
    stats = {
        "num_episodes": len(all_episodes),
        "reward_mean": sum(rewards) / max(len(rewards), 1),
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "success_rate": sum(1 for r in rewards if r > 0) / max(len(rewards), 1),
        "avg_tool_calls": sum(ep.num_tool_calls for ep in all_episodes) / max(len(all_episodes), 1),
    }
    return batch, stats


def _pad(tensors: list[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    """Pad list of 1D tensors to same length."""
    max_len = max(t.size(0) for t in tensors)
    out = torch.full((len(tensors), max_len), pad_value, dtype=torch.long)
    for i, t in enumerate(tensors):
        out[i, :t.size(0)] = t
    return out


def _pad_float(tensors: list[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    """Pad list of 1D float tensors to same length."""
    max_len = max(t.size(0) for t in tensors)
    out = torch.full((len(tensors), max_len), pad_value, dtype=torch.float32)
    for i, t in enumerate(tensors):
        out[i, :t.size(0)] = t
    return out
