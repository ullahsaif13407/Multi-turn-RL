# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Train Qwen (1.5B+) to use tools via GRPO reinforcement learning. Multi-turn episodes with mock tool environment.

## Commands

```bash
pip install -r requirements.txt
pytest tests/ -v
pytest tests/test_smoke.py::test_grpo_loss -v  # single test
python scripts/train.py
python scripts/train.py --config my_config.yaml
```

Test markers: `unit`, `integration`, `slow`, `gpu`, `docker`.

## Architecture

```
src/
  environment.py  — Environment, Action, Episode, mock tools (calculator, execute_code, read_file)
  policy.py       — QwenPolicy with LoRA via peft
  grpo.py         — GRPO loss, advantages, train_step (functional, no class)
  rollout.py      — collect_episode, collect_grpo_batch
scripts/
  train.py        — Training loop (yaml config, optional W&B)
config.yaml       — Single flat config
tests/
  test_smoke.py   — One test per module, no GPU needed
```

## Key Patterns

- GRPO is functional: `compute_log_probs` → `compute_advantages` (group-relative) → REINFORCE + KL penalty → `train_step`
- Environment follows reset/step API: `step(Action) → (obs, reward, done, info)`
- `GRPOBatch` dataclass holds all tensors for a training step
- Mock tools return deterministic results for testing

## Code Style

- Minimal, no bloat. Only add abstraction when there are 2+ users.
- `@dataclass` for data containers. Functional core algorithms.
- PyTorch + peft for LoRA fine-tuning.
