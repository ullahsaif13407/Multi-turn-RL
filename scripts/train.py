#!/usr/bin/env python3
"""Training loop for multi-turn GRPO.

Usage:
    python scripts/train.py                    # defaults from config.yaml
    python scripts/train.py --config my.yaml   # custom config
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment import Environment
from src.policy import QwenPolicy, PolicyConfig
from src.grpo import train_step as grpo_train_step
from src.rollout import collect_grpo_batch

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train(cfg: dict) -> None:
    m = cfg["model"]
    t = cfg["training"]
    r = cfg["rewards"]

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Policy
    policy_cfg = PolicyConfig(
        model_name=m["name"],
        max_length=m["max_length"],
        lora_r=m["lora_r"],
        lora_alpha=m["lora_alpha"],
    )
    policy = QwenPolicy(policy_cfg).load()
    logger.info(f"Loaded {m['name']}, trainable params: {sum(p.numel() for p in policy.trainable_parameters()):,}")

    # Reference model (frozen, for KL)
    ref_model = None
    if t["kl_coef"] > 0:
        from transformers import AutoModelForCausalLM
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        ref_model = AutoModelForCausalLM.from_pretrained(
            m["name"], torch_dtype=dtype_map.get(m.get("dtype", "bfloat16"), torch.bfloat16),
            device_map="auto", trust_remote_code=True,
        )
        for p in ref_model.parameters():
            p.requires_grad = False
        logger.info("Loaded reference model for KL")

    # Environment
    env = Environment(
        max_turns=t["max_turns"],
        turn_penalty=r["turn_penalty"],
        success_reward=r["success"],
        failure_reward=r["failure"],
    )

    # Optimizer
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=t["lr"])

    # W&B
    if HAS_WANDB and cfg.get("wandb", {}).get("enabled", False):
        wandb.init(project=cfg["wandb"].get("project", "multi-turn-rl"), config=cfg)

    # Training loop
    for step in range(t["num_iterations"]):
        batch, stats = collect_grpo_batch(
            policy=policy,
            env=env,
            tokenizer=policy.tokenizer,
            num_prompts=t["batch_size"],
            group_size=t["group_size"],
            max_turns=t["max_turns"],
            device=device,
        )

        loss_val, metrics = grpo_train_step(
            model=policy.model,
            optimizer=optimizer,
            batch=batch,
            ref_model=ref_model,
            kl_coef=t["kl_coef"],
        )

        logger.info(
            f"[{step}/{t['num_iterations']}] loss={loss_val:.4f} "
            f"reward={stats['reward_mean']:.3f} success={stats['success_rate']:.1%}"
        )

        if HAS_WANDB and wandb.run:
            wandb.log({"loss": loss_val, **stats, **metrics}, step=step)

    # Save
    output_dir = Path(cfg.get("output_dir", "experiments/latest"))
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.model.save_pretrained(str(output_dir / "policy"))
    policy.tokenizer.save_pretrained(str(output_dir / "policy"))
    logger.info(f"Saved to {output_dir}")

    if HAS_WANDB and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
