# Multi-turn_RL Continuity

## Goal
Train Qwen 3 (1.7B+) on tool use via GRPO reinforcement learning. Interleaved thinking + tool calling with verifiable rewards.

## State

### Done
- Rebuilt rollout pipeline for Qwen 3 native `<think>` + `<tool_call>` support
- `src/data.py` — Task dataclass + dataset loaders (GSM8K, MBPP)
- `src/rewards.py` — Verifiable rewards: gsm8k (numeric match), code_exec (subprocess), exact_match
- `src/environment.py` — Added `execute_tool()` method (delegates to mock tools)
- `src/policy.py` — Default model → Qwen/Qwen3-1.7B, added `generate_turn()` with `enable_thinking=True`, stops at `<|im_end|>`
- `src/grpo.py` — Added `loss_mask` field to GRPOBatch, plumbed through `compute_loss` (effective_mask = response_mask * loss_mask)
- `src/rollout.py` — Full rewrite: `parse_tool_call()`, `collect_episode()` (generate→tool→resume loop), `build_loss_mask()` (state machine: assistant=1, else=0), `collect_grpo_batch()` with loss masks
- `scripts/train.py` — Wired dataset loading, samples tasks per step, passes generation config
- `config.yaml` — Added `data` + `generation` sections
- `tests/test_smoke.py` — 26 tests (25 pass, 1 skip without transformers)

### Now
- Implementation complete. All tests passing.

### Next
- First training run: `python scripts/train.py` with GSM8K on GPU
- Add `datasets` to requirements.txt when ready to use real data
- Evaluate: check reward > 0 for some episodes, verify loss mask correctness on real tokenizer
- Consider: SFT bootstrap before GRPO, longer context for multi-step reasoning

## Key Decisions
| Decision | Choice |
|----------|--------|
| Model | Qwen/Qwen3-1.7B (native thinking + tool_call) |
| RL algo | GRPO (group-based, no learned baseline) |
| Fine-tuning | LoRA via peft |
| Config | YAML flat config |
| Token pattern | `<think>` + `<tool_call>` (Qwen 3 native) |
| Loss masking | assistant blocks = 1, system/user/tool = 0 |
| Rewards | GSM8K: numeric match, MBPP: code exec, generic: exact match |
| Sandbox | Mock tools (calculator, execute_code, read_file) |

## Architecture
```
query → <think> → (<think> → tool_call → tool_result)* → <think> → response
```
Loss mask: 1 for all tokens inside `<|im_start|>assistant...<|im_end|>`, 0 for everything else.
