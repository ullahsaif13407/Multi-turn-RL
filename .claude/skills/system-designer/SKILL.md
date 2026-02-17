---
name: system-designer
description: Design system architecture for RL frameworks, including component interfaces, data flow, and design patterns. Use when making architectural decisions, refactoring for modularity, or designing scalable RL systems.
---

# System Designer - RL Architecture & Design Patterns

Design robust, modular, and scalable architectures for RL environments and training systems.

## When to Use This Skill

- Designing overall system architecture
- Making framework/library choices
- Defining component interfaces and APIs
- Planning data flow and state management
- Choosing design patterns for extensibility
- Refactoring for better modularity
- Designing for testability and maintainability

## Core Design Principles for RL Systems

### 1. Modularity
**Separate concerns into independent components:**
- Environment logic ↔ Agent logic ↔ Training infrastructure
- Algorithm implementation ↔ Network architecture ↔ Training loop
- Configuration ↔ Execution ↔ Logging

**Benefits:**
- Easy to swap components (e.g., change algorithm without touching environment)
- Parallel development on different components
- Focused testing of individual modules
- Clear responsibility boundaries

### 2. Extensibility
**Design for future additions without breaking existing code:**
- Plugin system for new algorithms
- Registry pattern for environments
- Factory pattern for network architectures
- Strategy pattern for exploration methods

**Key Questions:**
- How easy is it to add a new RL algorithm?
- Can we support new observation types (images, graphs) without refactoring?
- Can we integrate new logging backends without code changes?

### 3. Reproducibility
**Ensure consistent results across runs:**
- Centralized seed management
- Configuration versioning
- Deterministic environment stepping
- Complete hyperparameter logging

**Implementation:**
- Config files checked into version control
- Automatic logging of all settings
- Seed propagation to all random number generators
- Checkpoint metadata includes config hash

### 4. Performance
**Optimize for training efficiency:**
- Vectorized environments (parallel rollouts)
- GPU utilization for neural networks
- Efficient replay buffer (circular buffer, not list)
- Avoid unnecessary data copies

**Design Patterns:**
- Lazy initialization for heavy objects
- Memory pooling for frequent allocations
- Batched operations for GPU efficiency
- Async data loading for parallel workers

### 5. Observability
**Make system behavior visible:**
- Comprehensive logging at all levels
- Real-time metric dashboards
- Debugging hooks (episode replay, state inspection)
- Performance profiling integration

## Component Architecture

### Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                     │
│  CLI, API, Dashboard (Streamlit/Gradio)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                  ORCHESTRATION LAYER                        │
│  Training Manager, Experiment Tracker, Config Loader       │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                     ALGORITHM LAYER                         │
│  PPO, DQN, SAC, A2C (Agent implementations)                 │
└────────┬──────────────────────────────────┬─────────────────┘
         │                                  │
┌────────┴────────────┐         ┌──────────┴──────────────┐
│  NETWORK LAYER      │         │  MEMORY LAYER           │
│  Policy, Value, Q   │         │  Replay Buffer, Rollout │
│  Encoders, Decoders │         │  Priority Queue         │
└─────────────────────┘         └─────────────────────────┘
         │                                  │
┌────────┴──────────────────────────────────┴─────────────────┐
│                    ENVIRONMENT LAYER                         │
│  BaseEnv, VectorizedEnv, Wrappers, Registry                 │
└──────────────────────────────────────────────────────────────┘
         │
┌────────┴──────────────────────────────────────────────────┐
│                    INFRASTRUCTURE LAYER                    │
│  Logging, Checkpointing, Metrics, Visualization           │
└────────────────────────────────────────────────────────────┘
```

### Component Interfaces

#### Environment Interface (Gymnasium API)

```python
class BaseEnvironment:
    """Base interface for all RL environments."""

    @property
    def observation_space(self) -> gym.Space:
        """Define observation space."""

    @property
    def action_space(self) -> gym.Space:
        """Define action space."""

    def reset(self, seed: Optional[int] = None) -> Tuple[Observation, dict]:
        """Reset environment to initial state."""

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """Execute action and return (obs, reward, terminated, truncated, info)."""

    def render(self) -> Optional[np.ndarray]:
        """Render current state."""

    def close(self):
        """Cleanup resources."""
```

#### Agent Interface

```python
class BaseAgent:
    """Base interface for all RL agents."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, config: dict):
        """Initialize agent with env spaces and configuration."""

    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action given observation."""

    def update(self, batch: dict) -> dict:
        """Update agent from batch of experience. Returns metrics."""

    def save(self, path: str):
        """Save agent state to disk."""

    def load(self, path: str):
        """Load agent state from disk."""

    def get_state(self) -> dict:
        """Return current agent state for checkpointing."""

    def set_state(self, state: dict):
        """Restore agent state from checkpoint."""
```

#### Training Manager Interface

```python
class TrainingManager:
    """Orchestrates training runs."""

    def __init__(self, config: TrainingConfig):
        """Initialize with training configuration."""

    def setup(self):
        """Setup environments, agent, loggers, etc."""

    def train(self, num_timesteps: int):
        """Run training for specified timesteps."""

    def evaluate(self, num_episodes: int) -> dict:
        """Evaluate current agent. Returns metrics."""

    def checkpoint(self):
        """Save checkpoint with agent, config, metrics."""

    def resume_from_checkpoint(self, path: str):
        """Resume training from checkpoint."""
```

## Design Patterns for RL Systems

### 1. Registry Pattern (Environments & Algorithms)

**Purpose:** Decouple environment/algorithm creation from usage

```python
# Environment registry
class EnvironmentRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str, env_class: Type[BaseEnvironment], **kwargs):
        cls._registry[name] = (env_class, kwargs)

    @classmethod
    def make(cls, name: str, **override_kwargs) -> BaseEnvironment:
        env_class, default_kwargs = cls._registry[name]
        kwargs = {**default_kwargs, **override_kwargs}
        return env_class(**kwargs)

# Usage
EnvironmentRegistry.register("CartPole-v1", GymEnvironment, gym_id="CartPole-v1")
env = EnvironmentRegistry.make("CartPole-v1")
```

### 2. Factory Pattern (Network Architectures)

**Purpose:** Create different network architectures from config

```python
class NetworkFactory:
    @staticmethod
    def create_policy_network(obs_space: gym.Space, action_space: gym.Space, config: dict):
        if config['architecture'] == 'mlp':
            return MLPPolicy(obs_space, action_space, config['hidden_sizes'])
        elif config['architecture'] == 'cnn':
            return CNNPolicy(obs_space, action_space, config['cnn_config'])
        elif config['architecture'] == 'attention':
            return AttentionPolicy(obs_space, action_space, config['attn_config'])
        else:
            raise ValueError(f"Unknown architecture: {config['architecture']}")
```

### 3. Strategy Pattern (Exploration)

**Purpose:** Swap exploration strategies without changing agent code

```python
class ExplorationStrategy(ABC):
    @abstractmethod
    def select_action(self, q_values: np.ndarray, step: int) -> int:
        pass

class EpsilonGreedy(ExplorationStrategy):
    def select_action(self, q_values: np.ndarray, step: int) -> int:
        epsilon = self.epsilon_schedule(step)
        if np.random.random() < epsilon:
            return np.random.choice(len(q_values))
        return np.argmax(q_values)

class Boltzmann(ExplorationStrategy):
    def select_action(self, q_values: np.ndarray, step: int) -> int:
        temperature = self.temperature_schedule(step)
        probs = softmax(q_values / temperature)
        return np.random.choice(len(q_values), p=probs)

# Agent uses strategy
class DQNAgent:
    def __init__(self, exploration_strategy: ExplorationStrategy):
        self.exploration = exploration_strategy

    def select_action(self, obs):
        q_values = self.q_network(obs)
        return self.exploration.select_action(q_values, self.step)
```

### 4. Wrapper Pattern (Environment Preprocessing)

**Purpose:** Add functionality to environments without modifying them

```python
class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations to zero mean, unit variance."""

    def __init__(self, env):
        super().__init__(env)
        self.obs_mean = np.zeros(env.observation_space.shape)
        self.obs_std = np.ones(env.observation_space.shape)
        self.count = 0

    def observation(self, obs):
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)

class FrameStack(gym.Wrapper):
    """Stack last N frames for history."""

    def __init__(self, env, num_frames=4):
        super().__init__(env)
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

# Usage
env = EnvironmentRegistry.make("Pong-v0")
env = NormalizeObservation(env)
env = FrameStack(env, num_frames=4)
```

### 5. Observer Pattern (Logging & Callbacks)

**Purpose:** Decouple training loop from logging/monitoring

```python
class TrainingCallback(ABC):
    @abstractmethod
    def on_episode_end(self, episode: int, metrics: dict):
        pass

    @abstractmethod
    def on_training_step(self, step: int, metrics: dict):
        pass

class TensorBoardLogger(TrainingCallback):
    def on_episode_end(self, episode: int, metrics: dict):
        self.writer.add_scalar('episode_reward', metrics['reward'], episode)

class WandBLogger(TrainingCallback):
    def on_episode_end(self, episode: int, metrics: dict):
        wandb.log({'episode_reward': metrics['reward']}, step=episode)

class CheckpointSaver(TrainingCallback):
    def on_training_step(self, step: int, metrics: dict):
        if step % self.checkpoint_freq == 0:
            self.save_checkpoint(step)

# Training loop uses callbacks
class Trainer:
    def __init__(self, callbacks: List[TrainingCallback]):
        self.callbacks = callbacks

    def train(self):
        for step in range(self.num_steps):
            metrics = self.train_step()
            for callback in self.callbacks:
                callback.on_training_step(step, metrics)
```

## Data Flow Architecture

### Training Data Flow

```
┌──────────────┐
│ Environment  │
│   (reset)    │
└──────┬───────┘
       │ observation
       ▼
┌──────────────┐
│    Agent     │
│ (select_act) │
└──────┬───────┘
       │ action
       ▼
┌──────────────┐
│ Environment  │
│   (step)     │
└──────┬───────┘
       │ (obs, reward, done, info)
       ▼
┌──────────────┐      ┌─────────────┐
│ Replay/      │─────▶│   Logger    │
│ Rollout      │      │ (metrics)   │
│ Buffer       │      └─────────────┘
└──────┬───────┘
       │ batch
       ▼
┌──────────────┐
│    Agent     │
│  (update)    │
└──────┬───────┘
       │ loss metrics
       ▼
┌──────────────┐
│   Logger     │
│ (TensorBoard)│
└──────────────┘
```

### Configuration Flow

```
YAML Config ──▶ Config Parser ──▶ TrainingConfig
                                        │
    ┌───────────────────────────────────┴──────────────────┐
    ▼                           ▼                          ▼
Environment Config        Agent Config            Training Config
    │                           │                          │
    ▼                           ▼                          ▼
Environment           Agent + Networks              Training Manager
```

## Decision Framework

When making architectural decisions, consider:

### 1. Framework Choice (PyTorch vs JAX)

**PyTorch:**
- ✅ Larger ecosystem, more tutorials
- ✅ Easier debugging (eager execution)
- ✅ Better for complex architectures
- ❌ Slower for simple operations
- ❌ Less functional programming support

**JAX/Flax:**
- ✅ Faster for vectorized ops
- ✅ Better for research/experimentation
- ✅ Functional programming benefits
- ❌ Steeper learning curve
- ❌ Smaller ecosystem

**Recommendation:** PyTorch for practical applications, JAX for research

### 2. Build vs. Integrate (Stable-Baselines3/RLlib)

**Build from Scratch:**
- ✅ Full control and customization
- ✅ Learning opportunity
- ✅ Optimized for specific use case
- ❌ More development time
- ❌ Potential bugs and edge cases

**Integrate Existing:**
- ✅ Battle-tested implementations
- ✅ Faster to production
- ✅ Community support
- ❌ Less flexibility
- ❌ Learning curve for library internals

**Recommendation:** Start with Stable-Baselines3, customize as needed

### 3. Experiment Tracking (TensorBoard vs W&B vs MLflow)

**TensorBoard:**
- ✅ Local, no cloud dependency
- ✅ Deep integration with PyTorch
- ✅ Free
- ❌ Limited collaboration features
- ❌ Manual organization

**Weights & Biases:**
- ✅ Excellent collaboration
- ✅ Automatic experiment organization
- ✅ Rich visualizations
- ❌ Cloud-based (data privacy concerns)
- ❌ Free tier limitations

**MLflow:**
- ✅ Self-hosted option
- ✅ Model registry built-in
- ✅ Good for production pipelines
- ❌ Less user-friendly UI
- ❌ Setup overhead

**Recommendation:** TensorBoard for local dev, W&B for team projects

## System Design Checklist

When designing a new component:

```
☐ Clear interface definition (input/output types)
☐ Configuration schema (YAML structure)
☐ Initialization and cleanup (resource management)
☐ Error handling strategy (fail fast vs. graceful degradation)
☐ Logging integration (what metrics to track)
☐ Testing approach (unit, integration, end-to-end)
☐ Documentation (docstrings, usage examples)
☐ Performance considerations (bottlenecks, optimization)
☐ Extensibility points (where to add new features)
☐ Backward compatibility (if refactoring existing code)
```

## Architecture Documentation Template

```markdown
# Component: [Name]

## Purpose
[What problem does this solve?]

## Interface
[Public API with types]

## Dependencies
[What components/libraries does this depend on?]

## Data Flow
[Diagram showing how data moves through component]

## Configuration
[YAML schema and defaults]

## Extension Points
[How can this be extended or customized?]

## Performance
[Expected performance characteristics, bottlenecks]

## Testing
[How to test this component]

## Example Usage
[Code example]
```

---

**Ready to design!** Provide thorough architectural guidance for building robust, maintainable, and scalable RL systems.
