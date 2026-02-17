---
name: coder
description: Implement RL environments, algorithms, and training infrastructure. Use when writing new code, implementing planned features, refactoring existing code, or fixing bugs in RL systems.
allowed-tools: Read, Grep, Glob, Edit, Write, Bash
---

# Coder - RL Implementation Specialist

Implement high-quality, well-tested code for RL environments, agents, and training infrastructure.

## When to Use This Skill

- Implementing new RL algorithms (PPO, DQN, SAC, etc.)
- Creating custom environments
- Building training infrastructure
- Refactoring existing code for better modularity
- Fixing bugs in RL systems
- Adding features to existing components
- Writing tests for RL code

## Coding Standards

### Python Best Practices

**Type Hints**
```python
from typing import Tuple, Optional, Dict, Any
import numpy as np
import gymnasium as gym

def select_action(
    observation: np.ndarray,
    deterministic: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Select action given observation.

    Args:
        observation: Environment observation
        deterministic: If True, select best action; else explore

    Returns:
        Tuple of (action, info_dict)
    """
    pass
```

**Docstrings (Google Style)**
```python
class PPOAgent:
    """Proximal Policy Optimization agent.

    PPO is an on-policy algorithm that uses clipped surrogate objective
    to prevent large policy updates.

    Attributes:
        observation_space: Observation space of environment
        action_space: Action space of environment
        policy_network: Actor network for policy
        value_network: Critic network for value estimation

    Example:
        >>> agent = PPOAgent(env.observation_space, env.action_space, config)
        >>> action, info = agent.select_action(obs)
        >>> metrics = agent.update(rollout_buffer)
    """
```

**Error Handling**
```python
class EnvironmentError(Exception):
    """Raised when environment is in invalid state."""
    pass

def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
    if not self.action_space.contains(action):
        raise ValueError(
            f"Action {action} is outside action space {self.action_space}"
        )

    try:
        next_state = self._transition(self.state, action)
    except Exception as e:
        raise EnvironmentError(f"Failed to transition state: {e}") from e

    return next_state, reward, terminated, truncated, info
```

**Configuration Management**
```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class PPOConfig:
    """Configuration for PPO algorithm."""
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
```

### Code Organization

**File Structure**
```
rl_framework/
├── envs/
│   ├── __init__.py
│   ├── base.py              # BaseEnvironment interface
│   ├── registry.py          # Environment registry
│   ├── wrappers/
│   │   ├── __init__.py
│   │   ├── normalize.py     # Observation normalization
│   │   ├── frame_stack.py   # Frame stacking
│   │   └── reward_shaping.py
│   └── custom/
│       ├── __init__.py
│       └── gridworld.py     # Custom environments
├── agents/
│   ├── __init__.py
│   ├── base.py              # BaseAgent interface
│   ├── ppo/
│   │   ├── __init__.py
│   │   ├── agent.py         # PPO agent
│   │   ├── networks.py      # Policy/value networks
│   │   └── buffer.py        # Rollout buffer
│   ├── dqn/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── networks.py
│   │   └── buffer.py        # Replay buffer
│   └── sac/
├── training/
│   ├── __init__.py
│   ├── trainer.py           # Training manager
│   ├── callbacks.py         # Training callbacks
│   └── evaluation.py        # Evaluation logic
├── utils/
│   ├── __init__.py
│   ├── logging.py           # Logging utilities
│   ├── seeding.py           # Seed management
│   └── schedule.py          # Learning rate schedules
├── configs/
│   ├── ppo_cartpole.yaml
│   ├── dqn_atari.yaml
│   └── sac_continuous.yaml
└── tests/
    ├── test_envs.py
    ├── test_agents.py
    └── test_training.py
```

## Implementation Patterns

### Pattern 1: Environment Implementation

```python
import gymnasium as gym
import numpy as np
from typing import Tuple, Optional, Dict, Any

class GridWorldEnv(gym.Env):
    """Simple grid world environment.

    Agent navigates grid to reach goal while avoiding obstacles.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        grid_size: int = 10,
        num_obstacles: int = 5,
        render_mode: Optional[str] = None
    ):
        super().__init__()

        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.render_mode = render_mode

        # Define spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(grid_size, grid_size, 3),  # Grid with 3 channels
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)  # Up, down, left, right

        # Initialize state
        self._agent_pos = None
        self._goal_pos = None
        self._obstacles = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Initialize positions
        self._agent_pos = self._random_empty_position()
        self._goal_pos = self._random_empty_position()
        self._obstacles = [
            self._random_empty_position() for _ in range(self.num_obstacles)
        ]

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and return result."""
        # Map action to direction
        direction = self._action_to_direction(action)

        # Compute new position
        new_pos = self._agent_pos + direction
        new_pos = np.clip(new_pos, 0, self.grid_size - 1)

        # Check for obstacles
        if not self._is_obstacle(new_pos):
            self._agent_pos = new_pos

        # Compute reward
        terminated = self._is_goal(self._agent_pos)
        truncated = False

        reward = 1.0 if terminated else -0.01  # Small step penalty

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Generate observation from current state."""
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)

        # Channel 0: Agent position
        obs[self._agent_pos[0], self._agent_pos[1], 0] = 1.0

        # Channel 1: Goal position
        obs[self._goal_pos[0], self._goal_pos[1], 1] = 1.0

        # Channel 2: Obstacles
        for obstacle in self._obstacles:
            obs[obstacle[0], obstacle[1], 2] = 1.0

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Return auxiliary information."""
        return {
            "agent_pos": self._agent_pos,
            "goal_pos": self._goal_pos,
            "distance_to_goal": np.linalg.norm(self._agent_pos - self._goal_pos)
        }
```

### Pattern 2: Agent Implementation (PPO)

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PPOConfig:
    """PPO configuration."""
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

class PPOAgent:
    """Proximal Policy Optimization agent."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        config: PPOConfig,
        device: str = "cpu"
    ):
        self.obs_space = observation_space
        self.action_space = action_space
        self.config = config
        self.device = device

        # Create networks
        self.policy = self._build_policy_network()
        self.value = self._build_value_network()

        # Optimizer
        params = list(self.policy.parameters()) + list(self.value.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config.learning_rate)

        # Buffer
        self.rollout_buffer = RolloutBuffer(
            buffer_size=config.n_steps,
            observation_space=observation_space,
            action_space=action_space
        )

        self.num_timesteps = 0

    def select_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select action given observation."""
        with torch.no_grad():
            obs_tensor = torch.as_tensor(observation).float().to(self.device)

            # Get action distribution
            action_dist = self.policy(obs_tensor)

            if deterministic:
                action = action_dist.mode()
            else:
                action = action_dist.sample()

            # Get value estimate
            value = self.value(obs_tensor)

            # Get log probability
            log_prob = action_dist.log_prob(action)

            info = {
                "value": value.cpu().numpy(),
                "log_prob": log_prob.cpu().numpy()
            }

            return action.cpu().numpy(), info

    def update(self, rollout_buffer: RolloutBuffer) -> Dict[str, float]:
        """Update policy and value networks."""
        # Compute advantages and returns
        advantages, returns = self._compute_advantages_and_returns(rollout_buffer)

        # Normalize advantages
        if self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "clip_fraction": 0.0,
            "explained_variance": 0.0
        }

        # Multiple epochs of updates
        for epoch in range(self.config.n_epochs):
            # Sample batches
            for batch in rollout_buffer.sample_batches(self.config.batch_size):
                # Compute policy loss
                policy_loss, clip_fraction, entropy = self._compute_policy_loss(
                    batch, advantages
                )

                # Compute value loss
                value_loss, explained_var = self._compute_value_loss(
                    batch, returns
                )

                # Total loss
                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    - self.config.ent_coef * entropy
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                # Track metrics
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["clip_fraction"] += clip_fraction
                metrics["explained_variance"] += explained_var

        # Average metrics
        num_updates = self.config.n_epochs * len(rollout_buffer) // self.config.batch_size
        for key in metrics:
            metrics[key] /= num_updates

        return metrics

    def _compute_policy_loss(
        self, batch: Dict, advantages: torch.Tensor
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """Compute clipped policy loss."""
        # Get current action distribution
        action_dist = self.policy(batch["observations"])
        log_probs = action_dist.log_prob(batch["actions"])

        # Compute ratio
        old_log_probs = batch["log_probs"]
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped surrogate loss
        adv = advantages[batch["indices"]]
        policy_loss_1 = adv * ratio
        policy_loss_2 = adv * torch.clamp(
            ratio,
            1 - self.config.clip_range,
            1 + self.config.clip_range
        )
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        # Clip fraction (for monitoring)
        clip_fraction = torch.mean(
            (torch.abs(ratio - 1) > self.config.clip_range).float()
        ).item()

        # Entropy bonus
        entropy = action_dist.entropy().mean()

        return policy_loss, clip_fraction, entropy
```

### Pattern 3: Training Loop

```python
class Trainer:
    """Training manager for RL agents."""

    def __init__(
        self,
        agent: BaseAgent,
        env: gym.Env,
        eval_env: Optional[gym.Env] = None,
        callbacks: Optional[List[Callback]] = None,
        logger: Optional[Logger] = None
    ):
        self.agent = agent
        self.env = env
        self.eval_env = eval_env or env
        self.callbacks = callbacks or []
        self.logger = logger

        self.num_timesteps = 0
        self.num_episodes = 0

    def train(
        self,
        total_timesteps: int,
        eval_freq: int = 10000,
        checkpoint_freq: int = 50000
    ):
        """Run training loop."""
        observation, info = self.env.reset()
        episode_reward = 0.0
        episode_length = 0

        while self.num_timesteps < total_timesteps:
            # Select action
            action, action_info = self.agent.select_action(observation)

            # Execute action
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store transition
            self.agent.store_transition(
                observation, action, reward, next_observation, done, action_info
            )

            episode_reward += reward
            episode_length += 1
            self.num_timesteps += 1

            # Update agent
            if self.agent.ready_to_update():
                update_metrics = self.agent.update()
                self.logger.log(update_metrics, step=self.num_timesteps)

                # Callbacks
                for callback in self.callbacks:
                    callback.on_update(self.num_timesteps, update_metrics)

            # Episode end
            if done:
                self.num_episodes += 1

                # Log episode metrics
                episode_metrics = {
                    "episode_reward": episode_reward,
                    "episode_length": episode_length
                }
                self.logger.log(episode_metrics, step=self.num_timesteps)

                # Callbacks
                for callback in self.callbacks:
                    callback.on_episode_end(self.num_episodes, episode_metrics)

                # Reset
                observation, info = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
            else:
                observation = next_observation

            # Evaluation
            if self.num_timesteps % eval_freq == 0:
                eval_metrics = self.evaluate(num_episodes=10)
                self.logger.log(eval_metrics, step=self.num_timesteps)

            # Checkpointing
            if self.num_timesteps % checkpoint_freq == 0:
                self.save_checkpoint(self.num_timesteps)

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate agent."""
        episode_rewards = []
        episode_lengths = []

        for _ in range(num_episodes):
            observation, info = self.eval_env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                action, _ = self.agent.select_action(observation, deterministic=True)
                observation, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        return {
            "eval/mean_reward": np.mean(episode_rewards),
            "eval/std_reward": np.std(episode_rewards),
            "eval/mean_length": np.mean(episode_lengths)
        }
```

## Testing Guidelines

### Unit Tests

```python
import pytest
import numpy as np
import gymnasium as gym

def test_environment_reset():
    """Test environment reset functionality."""
    env = GridWorldEnv(grid_size=10)
    obs, info = env.reset(seed=42)

    assert obs.shape == (10, 10, 3)
    assert obs.dtype == np.float32
    assert "agent_pos" in info
    assert "goal_pos" in info

def test_environment_step():
    """Test environment step functionality."""
    env = GridWorldEnv(grid_size=10)
    env.reset(seed=42)

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (10, 10, 3)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

def test_agent_action_selection():
    """Test agent action selection."""
    env = gym.make("CartPole-v1")
    config = PPOConfig()
    agent = PPOAgent(env.observation_space, env.action_space, config)

    obs, _ = env.reset()
    action, info = agent.select_action(obs)

    assert env.action_space.contains(action)
    assert "value" in info
    assert "log_prob" in info
```

## Code Quality Checklist

Before submitting code:

```
☐ Type hints on all function signatures
☐ Docstrings for all public functions/classes
☐ Error handling for edge cases
☐ Configuration via dataclass or config file
☐ Logging for important events
☐ Unit tests for core functionality
☐ Integration test with simple environment
☐ Code formatted with black/ruff
☐ No unused imports or variables
☐ Performance profiled for bottlenecks (if critical path)
```

## Common Pitfalls to Avoid

**1. Not Validating Environment**
```python
# BAD: No validation
env = CustomEnv()

# GOOD: Validate before training
from gymnasium.utils.env_checker import check_env
env = CustomEnv()
check_env(env)  # Raises errors if invalid
```

**2. Memory Leaks in Buffers**
```python
# BAD: List keeps growing
self.buffer = []
self.buffer.append(transition)  # Memory leak!

# GOOD: Fixed-size circular buffer
from collections import deque
self.buffer = deque(maxlen=10000)
self.buffer.append(transition)
```

**3. Forgetting to Normalize Observations**
```python
# BAD: Raw pixel values [0, 255]
obs = env.reset()
action = agent.select_action(obs)

# GOOD: Normalized [0, 1]
obs = env.reset()
obs = obs / 255.0
action = agent.select_action(obs)
```

**4. Not Seeding Properly**
```python
# BAD: Non-reproducible
env.reset()

# GOOD: Reproducible
env.reset(seed=42)
env.action_space.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

---

**Ready to code!** Implement clean, tested, production-quality RL code following best practices.
