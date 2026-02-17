---
name: ui-ux-designer
description: Design user interfaces for RL training dashboards, experiment management, and visualization tools. Use when planning UI layouts, designing user flows, or creating wireframes for RL applications.
---

# UI/UX Designer - RL Dashboard & Interface Design

Design intuitive, informative interfaces for RL training, monitoring, and experimentation.

## When to Use This Skill

- Designing training dashboard layouts
- Creating user flows for experiment management
- Planning data visualizations for RL metrics
- Designing environment configuration interfaces
- Creating wireframes for hyperparameter tuning
- Improving existing UI based on user feedback

## Core Principles for RL UI Design

### 1. Information Hierarchy

**Primary Information** (Always visible)
- Training status (running/paused/completed)
- Current timestep / Total timesteps
- Live training reward (episode and smoothed)
- Agent performance (success rate, episode length)

**Secondary Information** (Accessible, not always visible)
- Hyperparameter values
- Network architecture details
- Environment configuration
- Checkpoint history

**Tertiary Information** (On-demand)
- Detailed logs
- Episode replays
- Gradient statistics
- System resource usage

### 2. Real-Time Feedback

Users need immediate feedback on:
- Training progress (progress bar + ETA)
- Agent performance trends (live charts)
- System health (GPU/CPU usage, memory)
- Errors or warnings (prominent alerts)

### 3. Contextual Actions

Actions should be:
- **Contextual**: "Pause Training" button only when training is active
- **Reversible**: "Resume Training" after pause
- **Confirmed**: "Stop Training" requires confirmation (data loss)
- **Grouped**: Related actions together (Start/Pause/Stop)

### 4. Progressive Disclosure

Don't overwhelm users:
- Start with essentials (training status, reward curve)
- Expand sections on demand (detailed metrics, logs)
- Use tabs/accordion for complex information
- Provide "Advanced Settings" for expert users

## User Personas for RL Systems

### 1. Researcher/Student
**Goals:**
- Experiment with different algorithms
- Understand agent behavior
- Compare hyperparameter effects

**Needs:**
- Easy algorithm selection
- Clear metric visualizations
- Experiment comparison tools
- Episode replay capability

### 2. ML Engineer
**Goals:**
- Train production agents
- Monitor training stability
- Optimize hyperparameters
- Deploy trained models

**Needs:**
- Robust training pipeline
- Checkpoint management
- Performance profiling
- Model export functionality

### 3. Domain Expert (Non-ML)
**Goals:**
- Train agents for specific tasks
- Evaluate agent behavior
- Adjust reward functions

**Needs:**
- Simplified configuration
- Intuitive environment setup
- Behavior visualization
- Pre-built templates

## Key UI Components for RL Systems

### 1. Training Dashboard

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│ Training Status Bar                                         │
│ CartPole-v1 | PPO | Running | 45,000 / 100,000 steps       │
│ [Pause] [Stop] [Checkpoint]                      45% ■■■■▫▫▫│
└─────────────────────────────────────────────────────────────┘

┌──────────────────────────────────┬──────────────────────────┐
│ Episode Reward                   │ Episode Length           │
│                                  │                          │
│  500 ┤                      •    │  200 ┤            •      │
│      │                   •       │      │         •         │
│  250 ┤              •            │  100 ┤    •              │
│      │         •                 │      │ •                 │
│    0 ┼─────────────────────────  │    0 ┼──────────────────│
│      0     25k    50k    75k     │      0    25k   50k  75k│
└──────────────────────────────────┴──────────────────────────┘

┌──────────────────────────────────┬──────────────────────────┐
│ Policy Loss                      │ Value Loss               │
│ (Recent: 0.032, Smoothed: 0.041) │ (Recent: 0.089)          │
└──────────────────────────────────┴──────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Quick Stats                                                 │
│ Success Rate: 87% | Avg Reward: 421.3 | FPS: 2,341         │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**
- Real-time updating charts (1-5 second refresh)
- Smoothed lines with confidence intervals
- Tooltips on hover showing exact values
- Zoom/pan functionality for detailed inspection
- Export chart as image

### 2. Experiment Configuration

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│ Create New Experiment                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Experiment Name: [my_experiment_001____________]            │
│                                                             │
│ Environment:  [CartPole-v1 ▼]                               │
│               Classic control task: Balance pole on cart    │
│                                                             │
│ Algorithm:    [PPO ▼]                                       │
│               On-policy, works well for continuous control  │
│                                                             │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ Hyperparameters                          [Use Defaults]  │
│ │                                                         │ │
│ │ Learning Rate:    [0.0003___]                           │ │
│ │ Rollout Steps:    [2048_____]                           │ │
│ │ Batch Size:       [64_______]                           │ │
│ │ Discount (γ):     [0.99_____]                           │ │
│ │                                                         │ │
│ │ [▸ Advanced Settings]                                   │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Training Budget:                                            │
│ ○ Timesteps:  [100000__]                                    │
│ ○ Episodes:   [500_____]                                    │
│ ○ Wall Time:  [2_______] hours                              │
│                                                             │
│                          [Cancel]  [Start Training]         │
└─────────────────────────────────────────────────────────────┘
```

**Design Principles:**
- Sensible defaults pre-filled
- Contextual help text for each field
- Validation on input (immediate feedback)
- Templates for common scenarios
- "Advanced Settings" collapsed by default

### 3. Experiment Comparison

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│ Compare Experiments                        [+ Add Experiment]│
│                                                             │
│ Selected: [✓] exp_lr_0.001  [✓] exp_lr_0.0003  [ ] exp_lr_0│
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Episode Reward Comparison                                   │
│                                                             │
│  500 ┤                                                      │
│      │             ╱╲    ── exp_lr_0.001                    │
│  250 ┤        ╱───╯  ╲                                      │
│      │   ╱───╯        ╲╲  ··· exp_lr_0.0003                │
│    0 ┼────────────────────────────────────────────         │
│      0        25k         50k          75k                  │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────┬──────────────────────┬──────────────┐
│ Experiment           │ Final Reward         │ Timesteps    │
├──────────────────────┼──────────────────────┼──────────────┤
│ exp_lr_0.001         │ 487.2 ± 12.3         │ 52,000       │
│ exp_lr_0.0003        │ 421.5 ± 18.7         │ 68,000       │
│ exp_lr_0.0001        │ 312.4 ± 45.2         │ 95,000       │
└──────────────────────┴──────────────────────┴──────────────┘
```

**Key Features:**
- Select multiple experiments to compare
- Overlay plots with distinct colors/styles
- Statistical summary table
- Highlight best performing configuration

### 4. Agent Behavior Visualization

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│ Episode Replay: Episode #1234                  [← →]        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────┐   Timeline               │
│   │                             │   0:00 ████████░░░░░ 0:45 │
│   │     [Environment Render]    │                           │
│   │                             │   Frame: 142 / 1024       │
│   │         🚗 🏁              │                           │
│   │                             │   [⏮ ⏪ ⏯ ⏩ ⏭]         │
│   └─────────────────────────────┘                           │
│                                                             │
│   Observation:                    Action:                   │
│   Position:  [0.42, -0.18]        Steer: 0.32              │
│   Velocity:  [1.23, 0.05]         Accel: 0.87              │
│   Angle:     0.12 rad                                       │
│                                                             │
│   Q-Values:              Policy Distribution:               │
│   Action 0: ▪▪▪▪▪▪ 0.67  [████████████░░░░░] 78%          │
│   Action 1: ▪▪▪▪ 0.45     [███░░░░░░░░░░░░░] 15%          │
│   Action 2: ▪▪ 0.23       [█░░░░░░░░░░░░░░░] 7%           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**
- Video playback controls
- Step-by-step inspection
- Observation/action display
- Policy/value visualization
- Export episode as video

### 5. Hyperparameter Tuning Interface

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│ Hyperparameter Tuning                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Base Configuration: [ppo_default ▼]                         │
│                                                             │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ Parameters to Tune                                      │ │
│ │                                                         │ │
│ │ [✓] Learning Rate                                       │ │
│ │     Range: [1e-5____] to [1e-2____]  Scale: [Log ▼]    │ │
│ │                                                         │ │
│ │ [✓] Batch Size                                          │ │
│ │     Options: [32, 64, 128, 256]                         │ │
│ │                                                         │ │
│ │ [ ] Discount Factor                                     │ │
│ │     Range: [0.95___] to [0.999__]  Scale: [Linear ▼]   │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Search Strategy:  ○ Grid Search  ● Random Search           │
│                   ○ Bayesian Optimization                   │
│                                                             │
│ Number of Trials: [20______]                                │
│ Parallel Runs:    [4_______]                                │
│                                                             │
│                            [Cancel]  [Start Tuning]         │
└─────────────────────────────────────────────────────────────┘
```

## User Flows

### Flow 1: New User - First Training Run

```
1. Landing Page
   ↓
2. "Quick Start" Tutorial
   - Choose environment (pre-populated: CartPole)
   - Choose algorithm (pre-populated: PPO)
   - Click "Start Training"
   ↓
3. Training Dashboard
   - Watch live reward curve
   - See tooltips explaining metrics
   ↓
4. Training Complete
   - Success message
   - "View Results" button
   ↓
5. Results Page
   - Final reward
   - Success rate
   - "Watch Episode Replay" button
```

### Flow 2: Researcher - Experiment Comparison

```
1. Experiments List
   - See all past experiments
   - Filter by environment/algorithm
   ↓
2. Select 2-3 experiments
   - Checkboxes for multi-select
   ↓
3. Click "Compare"
   ↓
4. Comparison Dashboard
   - Overlaid reward curves
   - Statistical comparison table
   - Download results as CSV
```

### Flow 3: Engineer - Production Training

```
1. Create New Experiment
   - Load config from YAML
   - Override specific parameters
   ↓
2. Validate Configuration
   - Check environment setup
   - Verify resource availability
   ↓
3. Start Training
   - Checkpoint every 10k steps
   - Auto-save on errors
   ↓
4. Monitor Training
   - Check loss curves for divergence
   - View system resource usage
   ↓
5. Training Complete
   - Export model checkpoint
   - Generate training report
```

## Design Patterns

### Pattern 1: Status Indicators

```
Running:    [●] Training (45,000 / 100,000)  [Pause]
Paused:     [⏸] Training Paused              [Resume]
Completed:  [✓] Training Complete            [View Results]
Failed:     [✗] Training Failed              [View Logs]
Queued:     [⋯] Waiting for Resources        [Cancel]
```

### Pattern 2: Collapsible Sections

```
▸ Advanced Hyperparameters
  (Click to expand)

▾ Advanced Hyperparameters
  ├─ Entropy Coefficient: 0.01
  ├─ Value Function Coefficient: 0.5
  ├─ Max Gradient Norm: 0.5
  └─ GAE Lambda: 0.95
```

### Pattern 3: Inline Validation

```
Learning Rate: [0.1_____]  ⚠️ Warning: Value unusually high
                              Recommended: 0.0001 - 0.001

Batch Size:    [7_______]  ❌ Error: Must be power of 2
```

### Pattern 4: Smart Defaults

```
Environment: [CartPole-v1 ▼]

Algorithm:   [PPO ▼]           ℹ️ Recommended for CartPole

             Other options:
             - DQN (also suitable)
             - SAC (for continuous actions)
```

## Accessibility Considerations

- **Color Blindness**: Use patterns/textures in addition to colors
- **Screen Readers**: Proper ARIA labels on all interactive elements
- **Keyboard Navigation**: All actions accessible via keyboard
- **High Contrast Mode**: Ensure visibility with system theme
- **Text Size**: Respect user's font size preferences

## Responsive Design

### Desktop (>1200px)
- Multi-column layout
- Detailed visualizations
- All information visible

### Tablet (768px - 1200px)
- Two-column layout
- Slightly simplified charts
- Collapsible sidebars

### Mobile (<768px)
- Single column
- Tabbed interface for sections
- Simplified visualizations
- Essential metrics only

## Design Checklist

Before finalizing UI design:

```
☐ Clear visual hierarchy (primary info prominent)
☐ Consistent spacing and alignment
☐ Meaningful color usage (not decorative)
☐ Loading states for async operations
☐ Error states with actionable messages
☐ Empty states with clear next actions
☐ Tooltips for complex metrics
☐ Keyboard shortcuts for common actions
☐ Mobile-responsive layout
☐ Accessibility compliance (WCAG 2.1 AA)
```

## Wireframe Template

```markdown
# UI Component: [Name]

## Purpose
[What user task does this support?]

## Layout
[ASCII wireframe]

## Key Elements
1. Element 1 - Purpose
2. Element 2 - Purpose

## Interactions
- User action → System response

## States
- Default state
- Loading state
- Error state
- Success state

## Responsive Behavior
- Desktop: [Description]
- Tablet: [Description]
- Mobile: [Description]
```

---

**Ready to design!** Create intuitive, informative interfaces that help users train and understand RL agents effectively.
