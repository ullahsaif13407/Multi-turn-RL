"""GRPO (Group Relative Policy Optimization) trainer.

Group-based comparison: sample multiple outputs per prompt, rank by reward,
update policy to increase probability of higher-reward outputs.
No value network needed — uses group mean as baseline.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GRPOBatch:
    """Batch for GRPO training."""
    input_ids: torch.Tensor       # (B, input_len)
    attention_mask: torch.Tensor   # (B, input_len)
    response_ids: torch.Tensor     # (B, response_len)
    response_mask: torch.Tensor    # (B, response_len)
    rewards: torch.Tensor          # (B,)
    group_ids: torch.Tensor        # (B,)
    ref_log_probs: Optional[torch.Tensor] = None  # (B, response_len)
    loss_mask: Optional[torch.Tensor] = None       # (B, response_len) — 1 for policy tokens


def compute_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token log probs of response given input."""
    full_ids = torch.cat([input_ids, response_ids], dim=1)
    full_mask = torch.cat([attention_mask, response_mask], dim=1)

    with torch.set_grad_enabled(model.training):
        outputs = model(input_ids=full_ids, attention_mask=full_mask, return_dict=True)

    # Shift logits to align with response tokens
    logits = outputs.logits[:, input_ids.size(1) - 1:-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs * response_mask


def compute_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    clip: float = 10.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Group-relative advantages: reward - group_mean, optionally normalized."""
    rewards = torch.clamp(rewards, -clip, clip)
    advantages = torch.zeros_like(rewards)

    for gid in torch.unique(group_ids):
        mask = group_ids == gid
        group_r = rewards[mask]
        adv = group_r - group_r.mean()
        if normalize and adv.std() > 1e-8:
            adv = adv / (adv.std() + 1e-8)
        advantages[mask] = adv

    return advantages


def compute_loss(
    model: nn.Module,
    batch: GRPOBatch,
    ref_model: Optional[nn.Module] = None,
    kl_coef: float = 0.1,
    max_grad_norm: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Compute GRPO loss: REINFORCE + KL penalty.

    Returns (loss, metrics_dict).
    """
    # Policy log probs
    policy_lp = compute_log_probs(model, batch.input_ids, batch.attention_mask, batch.response_ids, batch.response_mask)

    # Reference log probs (for KL)
    ref_lp = batch.ref_log_probs
    if ref_lp is None and ref_model is not None:
        with torch.no_grad():
            ref_lp = compute_log_probs(ref_model, batch.input_ids, batch.attention_mask, batch.response_ids, batch.response_mask)

    # Advantages
    advantages = compute_advantages(batch.rewards, batch.group_ids)

    # Effective mask: response_mask * loss_mask (if present)
    effective_mask = batch.response_mask
    if batch.loss_mask is not None:
        effective_mask = effective_mask * batch.loss_mask

    # REINFORCE: -advantage * sum(log_prob)
    policy_lp_sum = (policy_lp * effective_mask).sum(dim=-1)
    policy_loss = -(advantages * policy_lp_sum).mean()

    # KL penalty
    kl_loss = torch.tensor(0.0, device=policy_loss.device)
    kl_mean = torch.tensor(0.0, device=policy_loss.device)
    if ref_lp is not None and kl_coef > 0:
        kl_per_token = ref_lp - policy_lp
        kl_per_sample = (kl_per_token * effective_mask).sum(-1) / (effective_mask.sum(-1) + 1e-8)
        kl_mean = kl_per_sample.mean()
        kl_loss = kl_coef * kl_mean

    total_loss = policy_loss + kl_loss

    metrics = {
        "policy_loss": policy_loss.item(),
        "kl_loss": kl_loss.item(),
        "kl_mean": kl_mean.item(),
        "advantages_mean": advantages.mean().item(),
        "advantages_std": advantages.std().item(),
        "rewards_mean": batch.rewards.mean().item(),
        "log_probs_mean": policy_lp_sum.mean().item(),
    }
    return total_loss, metrics


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: GRPOBatch,
    ref_model: Optional[nn.Module] = None,
    kl_coef: float = 0.1,
    max_grad_norm: float = 1.0,
) -> tuple[float, dict]:
    """One GRPO training step. Returns (loss_value, metrics)."""
    model.train()
    loss, metrics = compute_loss(model, batch, ref_model, kl_coef)

    loss.backward()
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    return loss.item(), metrics
