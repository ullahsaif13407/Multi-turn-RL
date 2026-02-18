# Multi-turn RL for LLM Tool-Use

Train Qwen 3 to use tools via Group Relative Policy Optimization (GRPO). The model learns interleaved thinking + tool calling through reinforcement learning with verifiable rewards.

## How It Works

```
User Query → <think>reasoning</think> → <tool_call>{calculator, 2+2}</tool_call>
           → tool result: 4
           → <think>got it</think> → "The answer is 4"
```

The model generates reasoning inside `<think>` tags, calls tools when needed, receives results, and produces a final answer. GRPO trains it by sampling multiple responses per prompt, scoring them with verifiable rewards, and reinforcing the better ones — no critic network needed.

## Architecture

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│  Dataset   │────▶│  Rollout   │────▶│    GRPO    │
│  (GSM8K /  │     │  (generate │     │  (train    │
│   MBPP)    │     │   + tools) │     │   step)    │
└────────────┘     └─────┬──────┘     └────────────┘
                         │
                    ┌────┴────┐
                    ▼         ▼
              ┌──────────┐ ┌──────────┐
              │  Policy  │ │  Environ │
              │  (Qwen3  │ │  (mock   │
              │  + LoRA) │ │  tools)  │
              └──────────┘ └──────────┘
```

| Module | File | Purpose |
|--------|------|---------|
| Data | `src/data.py` | Task dataclass, GSM8K/MBPP loaders |
| Rewards | `src/rewards.py` | Verifiable rewards (numeric match, code execution, exact match) |
| Environment | `src/environment.py` | Episode state, mock tool execution |
| Policy | `src/policy.py` | Qwen model with LoRA fine-tuning |
| GRPO | `src/grpo.py` | Functional GRPO: log probs → advantages → REINFORCE + KL |
| Rollout | `src/rollout.py` | Multi-turn rollout loop, loss mask state machine, batch assembly |
| Training | `scripts/train.py` | Training loop with YAML config, optional W&B logging |

## Quick Start

```bash
pip install -r requirements.txt

# Run tests (no GPU needed)
pytest tests/ -v

# Train
python scripts/train.py
python scripts/train.py --config my_config.yaml
```

## Configuration

All hyperparameters live in `config.yaml`:

```yaml
model:
  name: Qwen/Qwen3-1.7B
  lora_r: 16
  lora_alpha: 32

data:
  dataset: gsm8k        # gsm8k | mbpp
  max_samples: 1000

training:
  lr: 1.0e-5
  batch_size: 4          # tasks per step
  group_size: 4          # episodes per task
  kl_coef: 0.1
  num_iterations: 100
```

## Key Design Decisions

- **GRPO over PPO** — group-relative baselines eliminate the need for a value network. Sample multiple completions per prompt, use group mean reward as baseline.
- **Loss masking** — only train on assistant tokens. System prompts, user queries, and tool results are masked out via a state machine that scans `<|im_start|>role` boundaries.
- **Verifiable rewards** — no reward model. GSM8K checks numeric answers against ground truth. MBPP runs generated code against test assertions in a subprocess.
- **LoRA** — fine-tune ~0.5% of parameters (attention + MLP projections) to keep memory manageable on a single GPU.

## Reward Functions

| Type | Dataset | How |
|------|---------|-----|
| `gsm8k` | GSM8K | Extract last number from response, compare to ground truth |
| `code_exec` | MBPP | Run model's code + test assertions in subprocess, 1.0 if pass |
| `exact_match` | Any | Check if ground truth string appears in response |

## Token Pattern

Qwen 3 native special tokens:

```
<|im_start|>system\n...<|im_end|>
<|im_start|>user\n...<|im_end|>
<|im_start|>assistant\n
  <think>reasoning</think>
  <tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>
<|im_end|>
<|im_start|>tool\n{"output": "4"}<|im_end|>
<|im_start|>assistant\n
  <think>final reasoning</think>
  The answer is 4.
<|im_end|>
```
