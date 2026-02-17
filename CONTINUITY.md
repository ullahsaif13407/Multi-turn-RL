# Multi-turn_RL Continuity

## Goal
Train Qwen 3 (1.5B-14B) on tool use via GRPO reinforcement learning in sandboxed Docker environments.

## State

### Done
- Full implementation: environment, scheduler, trainer, policy, rollout
- Config system (Hydra): base, model (1.5B/7B/14B), algorithm (GRPO/PPO), environment configs
- Docker sandbox with container pooling
- SGLang inference client
- Test suite (8 test files)

### Now
- Ready to run tests and first training experiment

### Next
- `pip install -r requirements.txt`
- `pytest tests/ -v`
- First training run on Qwen 1.5B

## Key Decisions
| Decision | Choice |
|----------|--------|
| Inference | SGLang (continuous batching, RadixAttention) |
| RL algo | GRPO (group-based, no learned baseline) |
| Fine-tuning | LoRA via peft |
| Config | Hydra/OmegaConf |
| Sandbox | Docker containers with pooling |

## Open Questions
- UNCONFIRMED: GPU specs (model, VRAM)
- UNCONFIRMED: Benchmark tasks for evaluation
