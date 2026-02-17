---
name: ui-ux-coder
description: Implement user interfaces for RL dashboards using Streamlit or Gradio. Use when building training dashboards, experiment management UIs, or visualization tools for RL systems.
allowed-tools: Read, Grep, Glob, Edit, Write, Bash
---

# UI/UX Coder - RL Dashboard Implementation

Implement interactive, real-time dashboards for RL training and experimentation using Streamlit or Gradio.

## When to Use This Skill

- Building training monitoring dashboards
- Creating experiment configuration interfaces
- Implementing visualization tools
- Adding interactive controls to RL systems
- Integrating real-time metrics display
- Building hyperparameter tuning interfaces

## Framework Selection

### Streamlit
**Best for:**
- Rapid prototyping
- Data science workflows
- Internal tools
- Simple interactions

**Pros:**
- Minimal boilerplate code
- Built-in widgets and layouts
- Easy deployment
- Good documentation

**Cons:**
- Less control over layout
- Can be slow with large updates
- Limited customization

### Gradio
**Best for:**
- Model demos
- Public-facing interfaces
- Sharing with non-technical users
- Quick model testing

**Pros:**
- Very simple API
- Auto-generates shareable links
- Great for ML model interfaces
- Mobile-friendly

**Cons:**
- Less flexible layouts
- Fewer widgets
- Primarily for inference, not training monitoring

**Recommendation:** Use Streamlit for RL training dashboards (better for real-time updates and complex layouts)

## Streamlit Implementation Patterns

### Pattern 1: Real-Time Training Dashboard

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import time
import json

def load_training_metrics(log_file: Path) -> pd.DataFrame:
    """Load training metrics from log file."""
    metrics = []
    with open(log_file, 'r') as f:
        for line in f:
            metrics.append(json.loads(line))
    return pd.DataFrame(metrics)

def create_reward_plot(df: pd.DataFrame) -> go.Figure:
    """Create interactive reward plot."""
    fig = go.Figure()

    # Raw rewards
    fig.add_trace(go.Scatter(
        x=df['timestep'],
        y=df['episode_reward'],
        mode='lines',
        name='Episode Reward',
        line=dict(color='lightblue', width=1),
        opacity=0.5
    ))

    # Smoothed rewards
    smoothed = df['episode_reward'].rolling(window=10, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=df['timestep'],
        y=smoothed,
        mode='lines',
        name='Smoothed (10 episodes)',
        line=dict(color='blue', width=2)
    ))

    fig.update_layout(
        title='Training Reward',
        xaxis_title='Timestep',
        yaxis_title='Episode Reward',
        hovermode='x unified',
        template='plotly_white'
    )

    return fig

def main():
    st.set_page_config(
        page_title="RL Training Dashboard",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 RL Training Dashboard")

    # Sidebar for experiment selection
    with st.sidebar:
        st.header("Experiment Selection")

        # List available experiments
        experiments_dir = Path("experiments")
        experiments = [d.name for d in experiments_dir.iterdir() if d.is_dir()]

        selected_exp = st.selectbox("Select Experiment", experiments)

        # Training controls
        st.header("Training Controls")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏸ Pause", use_container_width=True):
                # Signal training to pause
                (experiments_dir / selected_exp / "pause.flag").touch()
                st.success("Training paused")

        with col2:
            if st.button("⏹ Stop", use_container_width=True):
                if st.session_state.get('confirm_stop', False):
                    (experiments_dir / selected_exp / "stop.flag").touch()
                    st.success("Training stopped")
                    st.session_state['confirm_stop'] = False
                else:
                    st.session_state['confirm_stop'] = True
                    st.warning("Click again to confirm")

        # Auto-refresh
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 1, 30, 5)

    # Main content area
    exp_path = experiments_dir / selected_exp
    log_file = exp_path / "training.log"

    if not log_file.exists():
        st.error(f"No training log found for {selected_exp}")
        return

    # Load metrics
    df = load_training_metrics(log_file)

    # Status bar
    latest = df.iloc[-1]
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)

    with status_col1:
        st.metric(
            "Timesteps",
            f"{latest['timestep']:,}",
            delta=None
        )

    with status_col2:
        st.metric(
            "Episode Reward",
            f"{latest['episode_reward']:.1f}",
            delta=f"{latest['episode_reward'] - df.iloc[-2]['episode_reward']:.1f}"
        )

    with status_col3:
        st.metric(
            "Success Rate",
            f"{latest.get('success_rate', 0) * 100:.1f}%"
        )

    with status_col4:
        fps = latest.get('fps', 0)
        st.metric("FPS", f"{fps:,.0f}")

    # Progress bar
    total_timesteps = latest.get('total_timesteps', 100000)
    progress = latest['timestep'] / total_timesteps
    st.progress(progress, text=f"{progress * 100:.1f}% complete")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Training Curves",
        "📊 Statistics",
        "🎮 Episode Replay",
        "⚙️ Configuration"
    ])

    with tab1:
        # Reward plot
        st.plotly_chart(
            create_reward_plot(df),
            use_container_width=True
        )

        # Loss plots in two columns
        col1, col2 = st.columns(2)

        with col1:
            if 'policy_loss' in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['timestep'],
                    y=df['policy_loss'],
                    mode='lines',
                    name='Policy Loss'
                ))
                fig.update_layout(
                    title='Policy Loss',
                    xaxis_title='Timestep',
                    yaxis_title='Loss',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'value_loss' in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['timestep'],
                    y=df['value_loss'],
                    mode='lines',
                    name='Value Loss',
                    line=dict(color='orange')
                ))
                fig.update_layout(
                    title='Value Loss',
                    xaxis_title='Timestep',
                    yaxis_title='Loss',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Statistics table
        st.subheader("Training Statistics")

        stats = {
            "Total Episodes": df['episode'].max() if 'episode' in df.columns else "N/A",
            "Total Timesteps": f"{latest['timestep']:,}",
            "Mean Reward (last 100)": f"{df['episode_reward'].tail(100).mean():.2f}",
            "Std Reward (last 100)": f"{df['episode_reward'].tail(100).std():.2f}",
            "Max Reward": f"{df['episode_reward'].max():.2f}",
            "Min Reward": f"{df['episode_reward'].min():.2f}",
        }

        stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # Distribution plot
        st.subheader("Reward Distribution (last 1000 episodes)")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['episode_reward'].tail(1000),
            nbinsx=50,
            name='Episode Rewards'
        ))
        fig.update_layout(
            xaxis_title='Reward',
            yaxis_title='Count',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("🎮 Episode Replay")

        # Episode selector
        if 'episode' in df.columns:
            episode_num = st.number_input(
                "Episode Number",
                min_value=1,
                max_value=int(df['episode'].max()),
                value=int(df['episode'].max())
            )

            # Check for saved episode video/frames
            video_path = exp_path / "videos" / f"episode_{episode_num}.mp4"
            if video_path.exists():
                st.video(str(video_path))
            else:
                st.info("No replay available for this episode. Enable video recording in config.")
        else:
            st.info("Episode data not available")

    with tab4:
        st.subheader("⚙️ Experiment Configuration")

        # Load and display config
        config_path = exp_path / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = f.read()
            st.code(config, language='yaml')

            # Download button
            st.download_button(
                "Download Config",
                config,
                file_name=f"{selected_exp}_config.yaml",
                mime="text/yaml"
            )
        else:
            st.warning("No configuration file found")

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
```

### Pattern 2: Experiment Configuration UI

```python
import streamlit as st
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class ExperimentConfig:
    """Configuration for RL experiment."""
    name: str
    environment: str
    algorithm: str
    learning_rate: float
    total_timesteps: int
    n_steps: int
    batch_size: int
    gamma: float
    seed: Optional[int] = None

def get_algorithm_info(algorithm: str) -> dict:
    """Get information about algorithm."""
    info = {
        "PPO": {
            "description": "Proximal Policy Optimization. Good for continuous control.",
            "default_lr": 3e-4,
            "hyperparams": ["n_steps", "batch_size", "n_epochs", "clip_range"]
        },
        "DQN": {
            "description": "Deep Q-Network. For discrete action spaces.",
            "default_lr": 1e-4,
            "hyperparams": ["buffer_size", "batch_size", "target_update_freq"]
        },
        "SAC": {
            "description": "Soft Actor-Critic. For continuous actions with exploration.",
            "default_lr": 3e-4,
            "hyperparams": ["buffer_size", "batch_size", "tau", "ent_coef"]
        }
    }
    return info.get(algorithm, {})

def create_experiment_form():
    """Create form for new experiment."""
    st.header("Create New Experiment")

    with st.form("experiment_form"):
        # Basic settings
        st.subheader("Basic Settings")

        name = st.text_input(
            "Experiment Name",
            value=f"exp_{int(time.time())}",
            help="Unique name for this experiment"
        )

        col1, col2 = st.columns(2)

        with col1:
            environment = st.selectbox(
                "Environment",
                ["CartPole-v1", "LunarLander-v2", "MountainCar-v0", "Pendulum-v1"],
                help="RL environment to train on"
            )

        with col2:
            algorithm = st.selectbox(
                "Algorithm",
                ["PPO", "DQN", "SAC"],
                help="RL algorithm to use"
            )

        # Show algorithm info
        algo_info = get_algorithm_info(algorithm)
        if algo_info:
            st.info(f"ℹ️ {algo_info['description']}")

        # Hyperparameters
        st.subheader("Hyperparameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            learning_rate = st.number_input(
                "Learning Rate",
                value=algo_info.get('default_lr', 3e-4),
                format="%.2e",
                help="Step size for gradient updates"
            )

        with col2:
            total_timesteps = st.number_input(
                "Total Timesteps",
                value=100000,
                step=10000,
                help="Total number of environment steps"
            )

        with col3:
            seed = st.number_input(
                "Random Seed",
                value=42,
                help="For reproducibility"
            )

        # Algorithm-specific hyperparameters
        with st.expander("Advanced Hyperparameters"):
            if algorithm == "PPO":
                col1, col2 = st.columns(2)
                with col1:
                    n_steps = st.number_input("Rollout Steps", value=2048)
                    batch_size = st.number_input("Batch Size", value=64)
                with col2:
                    n_epochs = st.number_input("Epochs per Update", value=10)
                    gamma = st.slider("Discount Factor (γ)", 0.9, 0.999, 0.99)

            elif algorithm == "DQN":
                col1, col2 = st.columns(2)
                with col1:
                    buffer_size = st.number_input("Buffer Size", value=100000)
                    batch_size = st.number_input("Batch Size", value=32)
                with col2:
                    target_update_freq = st.number_input("Target Update Freq", value=1000)
                    gamma = st.slider("Discount Factor (γ)", 0.9, 0.999, 0.99)

        # Submit button
        submitted = st.form_submit_button("🚀 Start Training", use_container_width=True)

        if submitted:
            # Create config
            config = ExperimentConfig(
                name=name,
                environment=environment,
                algorithm=algorithm,
                learning_rate=learning_rate,
                total_timesteps=total_timesteps,
                n_steps=n_steps if algorithm == "PPO" else 1,
                batch_size=batch_size,
                gamma=gamma,
                seed=seed
            )

            # Save config
            config_dir = Path("experiments") / name
            config_dir.mkdir(parents=True, exist_ok=True)

            config_path = config_dir / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(asdict(config), f, default_flow_style=False)

            st.success(f"✅ Experiment '{name}' created!")
            st.info("Navigate to Training Dashboard to monitor progress")

            return config

    return None
```

### Pattern 3: Experiment Comparison

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

def compare_experiments():
    """Compare multiple experiments."""
    st.header("📊 Experiment Comparison")

    # Load available experiments
    experiments_dir = Path("experiments")
    experiments = [d.name for d in experiments_dir.iterdir() if d.is_dir()]

    # Multi-select for experiments
    selected_experiments = st.multiselect(
        "Select Experiments to Compare",
        experiments,
        default=experiments[:min(3, len(experiments))]
    )

    if len(selected_experiments) < 2:
        st.warning("Select at least 2 experiments to compare")
        return

    # Load metrics for selected experiments
    all_metrics = {}
    for exp_name in selected_experiments:
        log_file = experiments_dir / exp_name / "training.log"
        if log_file.exists():
            df = load_training_metrics(log_file)
            all_metrics[exp_name] = df

    # Plot comparison
    st.subheader("Reward Comparison")

    fig = go.Figure()

    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i, (exp_name, df) in enumerate(all_metrics.items()):
        # Smoothed rewards
        smoothed = df['episode_reward'].rolling(window=10, min_periods=1).mean()

        fig.add_trace(go.Scatter(
            x=df['timestep'],
            y=smoothed,
            mode='lines',
            name=exp_name,
            line=dict(color=colors[i % len(colors)], width=2)
        ))

    fig.update_layout(
        title='Training Reward Comparison',
        xaxis_title='Timestep',
        yaxis_title='Episode Reward (smoothed)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Statistics table
    st.subheader("Performance Summary")

    summary_data = []
    for exp_name, df in all_metrics.items():
        summary_data.append({
            "Experiment": exp_name,
            "Final Reward": f"{df['episode_reward'].tail(10).mean():.2f}",
            "Max Reward": f"{df['episode_reward'].max():.2f}",
            "Timesteps": f"{df['timestep'].max():,}",
            "Success Rate": f"{df.get('success_rate', [0])[-1] * 100:.1f}%"
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Download comparison data
    if st.button("📥 Download Comparison Data"):
        csv = summary_df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            file_name="experiment_comparison.csv",
            mime="text/csv"
        )
```

## Streamlit Best Practices

### 1. Session State Management

```python
# Initialize session state
if 'training_active' not in st.session_state:
    st.session_state['training_active'] = False
    st.session_state['selected_experiment'] = None

# Use session state
if st.button("Start Training"):
    st.session_state['training_active'] = True
    st.session_state['selected_experiment'] = experiment_name

if st.session_state['training_active']:
    st.info("Training in progress...")
```

### 2. Caching for Performance

```python
@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_training_metrics(log_file: Path) -> pd.DataFrame:
    """Load and cache training metrics."""
    # Expensive operation
    return pd.read_csv(log_file)

@st.cache_resource  # Cache indefinitely
def load_model(model_path: str):
    """Load and cache trained model."""
    return torch.load(model_path)
```

### 3. Progress Indicators

```python
# For determinate progress
progress_bar = st.progress(0)
for i in range(100):
    time.sleep(0.01)
    progress_bar.progress(i + 1)

# For indeterminate progress
with st.spinner("Training in progress..."):
    train_agent()
```

### 4. Error Handling

```python
try:
    df = load_training_metrics(log_file)
except FileNotFoundError:
    st.error("Training log not found. Has training started?")
    st.stop()
except Exception as e:
    st.error(f"Error loading metrics: {e}")
    st.stop()
```

## UI Testing Checklist

Before deploying UI:

```
☐ All buttons/inputs have clear labels
☐ Loading states for async operations
☐ Error messages are actionable
☐ Tooltips on complex features
☐ Keyboard navigation works
☐ Mobile layout is readable
☐ Auto-refresh doesn't cause flickering
☐ Large datasets don't freeze UI
☐ Empty states have clear next actions
☐ Success/error messages are visible
```

## Common Pitfalls

**1. Too Many Reruns**
```python
# BAD: Causes infinite rerun loop
if st.button("Click"):
    st.session_state['count'] += 1
    st.rerun()  # Don't do this!

# GOOD: Session state updates automatically trigger rerun
if st.button("Click"):
    st.session_state['count'] += 1
```

**2. Slow Data Loading**
```python
# BAD: Reloads data on every interaction
df = pd.read_csv("large_file.csv")

# GOOD: Cache data
@st.cache_data
def load_data():
    return pd.read_csv("large_file.csv")
df = load_data()
```

**3. Not Using Containers**
```python
# BAD: Elements added in order
st.write("Footer")
st.write("Header")  # Can't reorder

# GOOD: Use containers
header = st.container()
footer = st.container()

with header:
    st.write("Header")

with footer:
    st.write("Footer")
```

---

**Ready to implement!** Build interactive, real-time dashboards for RL training and experimentation with clean, performant code.
