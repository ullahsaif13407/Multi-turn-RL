# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Train Qwen 3 (1.7B+) to use tools via GRPO reinforcement learning. Interleaved thinking + tool calling with verifiable rewards (GSM8K, MBPP).

## Commands

```bash
pip install -r requirements.txt
pytest tests/ -v
pytest tests/test_smoke.py::test_parse_tool_call_valid -v  # single test
python scripts/train.py
python scripts/train.py --config my_config.yaml
```

Test markers: `unit`, `integration`, `slow`, `gpu`, `docker`.

## Architecture

```
src/
  data.py         — Task dataclass, dataset loaders (GSM8K, MBPP)
  rewards.py      — Verifiable reward functions (gsm8k, code_exec, exact_match)
  environment.py  — Environment, Action, Episode, mock tools, execute_tool()
  policy.py       — QwenPolicy with LoRA, generate_turn() with thinking + tool_call
  grpo.py         — GRPO loss, advantages, train_step (functional, no class)
  rollout.py      — parse_tool_call(), collect_episode(), build_loss_mask(), collect_grpo_batch()
scripts/
  train.py        — Training loop (dataset-driven, yaml config, optional W&B)
config.yaml       — Model, data, generation, training, rewards config
tests/
  test_smoke.py   — 26 tests, no GPU needed
```

## Token Pattern (Qwen 3 native)

```
query → <think> → (<think> → tool_call → tool_result)* → <think> → response
```

Loss mask: 1 for `<|im_start|>assistant...<|im_end|>` blocks, 0 for system/user/tool.

## Key Patterns

- GRPO is functional: `compute_log_probs` → `compute_advantages` (group-relative) → REINFORCE + KL penalty → `train_step`
- Rollout loop: `generate_turn()` → `parse_tool_call()` → `execute_tool()` → repeat until no tool call
- `GRPOBatch` dataclass holds all tensors + `loss_mask` for a training step
- `build_loss_mask()` state machine: scans token IDs for `<|im_start|>role` boundaries
- Dataset-driven: `Task` dataclass with `ground_truth` / `test_code` for verifiable rewards
- Mock tools return deterministic results for testing

## Code Style

- Minimal, no bloat. Only add abstraction when there are 2+ users.
- `@dataclass` for data containers. Functional core algorithms.
- PyTorch + peft for LoRA fine-tuning.

## Skills

`.claude/skills/` provides automation slash commands:
- `/system-designer` — architecture design
- `/coder` — implementation
- `/deep-planner` — planning
- `/ui-ux-designer` / `/ui-ux-coder` — dashboards
- `/code-explainer` — ASCII architecture diagrams
- `/skill-creator` — create new skills
- `/research` — web + codebase research workflow
- `/post-x` — draft X.com posts about project results
