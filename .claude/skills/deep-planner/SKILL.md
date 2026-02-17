---
name: deep-planner
description: Deep planning for RL projects, features, and experiments. Use when starting new projects, planning major features, designing experiments, or breaking down complex ML/RL tasks into actionable steps.
context: fork
agent: Plan
---

# Deep Planner - RL Environment & Training Framework

Plan RL projects, features, and experiments with thorough analysis and step-by-step execution plans.

## When to Use This Skill

- Starting a new RL project or major feature
- Planning experiment design and evaluation strategy
- Breaking down complex RL algorithms into implementation steps
- Designing environment architecture or agent training pipeline
- Refactoring or restructuring existing RL codebase
- Planning integration of new RL algorithms or tools

## Planning Process

### Phase 1: Context Gathering

**Understand the Request**
1. What is the goal? (Train agent, build environment, implement algorithm, etc.)
2. What constraints exist? (Time, compute, existing codebase, dependencies)
3. What success criteria define completion?
4. Any specific requirements? (Performance targets, compatibility, API design)

**Analyze Current State**
1. Review existing codebase structure
2. Identify relevant files and components
3. Check for existing implementations or patterns
4. Understand dependencies and integrations

**Research Domain**
1. RL algorithm requirements (on-policy/off-policy, discrete/continuous actions)
2. Environment interface requirements (Gym/Gymnasium API)
3. Neural network architecture considerations
4. Training infrastructure needs (parallel envs, replay buffers, logging)

### Phase 2: Technical Design

**Architecture Decisions**
- Component boundaries and interfaces
- Data flow between components
- State management and persistence
- Error handling and recovery

**Implementation Approach**
- Build from scratch vs. integrate existing libraries
- Modular vs. monolithic structure
- Abstraction layers and extensibility points
- Testing strategy (unit, integration, end-to-end)

**Dependencies & Tools**
- Core libraries (PyTorch/JAX, Gymnasium, NumPy)
- Logging/visualization (TensorBoard, W&B, Matplotlib)
- Configuration management (Hydra, OmegaConf, YAML)
- Parallel processing (Ray, multiprocessing, joblib)

**Performance Considerations**
- Vectorized environments for parallel rollouts
- GPU utilization and memory management
- Efficient replay buffer implementation
- Checkpointing and resumption strategy

### Phase 3: Execution Plan

**Break Down into Steps**
1. Create detailed, sequential implementation steps
2. Identify critical path dependencies
3. Plan for incremental testing and validation
4. Define milestones and checkpoints

**Risk Mitigation**
- Identify potential blockers or challenges
- Plan for debugging and troubleshooting
- Consider edge cases and failure modes
- Build in validation checkpoints

**Documentation Requirements**
- API documentation for new components
- Configuration file schemas
- Usage examples and tutorials
- Architecture diagrams (if complex)

## Planning Templates

### New RL Algorithm Implementation

```
1. Research & Specification
   - Algorithm paper review
   - Pseudocode extraction
   - Hyperparameter defaults
   - Expected performance on benchmarks

2. Component Design
   - Network architectures (policy, value, Q-network)
   - Buffer/memory requirements (replay buffer, rollout buffer)
   - Loss functions and optimization
   - Exploration strategy

3. Implementation Steps
   - Base agent class and interface
   - Network modules (actor, critic, encoders)
   - Training loop structure
   - Checkpoint/resume functionality

4. Testing Strategy
   - Unit tests for components
   - Integration test with simple env (CartPole)
   - Benchmark against known results
   - Ablation studies

5. Documentation
   - API reference
   - Usage examples
   - Hyperparameter guide
   - Performance benchmarks
```

### New Environment Creation

```
1. Environment Specification
   - Observation space (shape, dtype, bounds)
   - Action space (discrete/continuous, bounds)
   - Reward structure (dense/sparse, range)
   - Episode termination conditions
   - Success criteria

2. Dynamics Implementation
   - State transition logic
   - Physics/rules simulation
   - Collision detection (if applicable)
   - Rendering (optional)

3. Validation & Testing
   - Space checks (observation/action validity)
   - Episode length tests
   - Reward range validation
   - Determinism checks (with seeding)
   - Gym/Gymnasium API compliance

4. Wrappers & Utilities
   - Observation preprocessing
   - Reward shaping (if needed)
   - Action rescaling/clipping
   - Frame stacking (for visual obs)

5. Registration & Documentation
   - Add to environment registry
   - Document observation/action spaces
   - Provide usage examples
   - Baseline performance metrics
```

### Training Pipeline Setup

```
1. Environment Setup
   - Choose/configure environment
   - Apply necessary wrappers
   - Vectorize for parallel rollouts
   - Validate environment correctness

2. Agent Configuration
   - Select RL algorithm
   - Design network architecture
   - Set hyperparameters
   - Initialize from checkpoint (optional)

3. Training Infrastructure
   - Setup logging (TensorBoard/W&B)
   - Configure checkpointing
   - Setup evaluation protocol
   - Resource allocation (GPU/CPU)

4. Training Loop
   - Implement training iteration
   - Periodic evaluation episodes
   - Metric tracking and logging
   - Checkpoint saving

5. Monitoring & Debugging
   - Training curve visualization
   - Performance metrics dashboard
   - Debugging tools (episode replay, attention viz)
   - Early stopping criteria
```

### Experiment Design

```
1. Hypothesis & Goals
   - Research question
   - Variables to test
   - Expected outcomes
   - Success metrics

2. Experimental Setup
   - Baseline configuration
   - Ablation variants
   - Environment selection
   - Evaluation protocol

3. Resource Planning
   - Compute requirements (GPU hours)
   - Storage for checkpoints/logs
   - Expected training time
   - Parallelization strategy

4. Execution Plan
   - Run baseline first
   - Sequential or parallel runs
   - Checkpointing strategy
   - Result aggregation

5. Analysis & Reporting
   - Metric comparison tables
   - Statistical significance tests
   - Learning curve plots
   - Findings summary
```

## Best Practices for RL Planning

### DO:
- ✅ Start with simple baselines (random policy, simple algorithm)
- ✅ Validate environments before training agents
- ✅ Use small-scale tests before full training runs
- ✅ Plan for reproducibility (seeding, configuration logging)
- ✅ Build modular components with clear interfaces
- ✅ Include evaluation metrics from the start
- ✅ Plan for checkpointing and training resumption
- ✅ Consider computational cost and optimization early

### DON'T:
- ❌ Skip environment validation (broken envs waste compute)
- ❌ Implement complex features without testing simple versions first
- ❌ Ignore reproducibility concerns until later
- ❌ Over-engineer abstractions before understanding requirements
- ❌ Train on full-scale problems without prototyping
- ❌ Forget about memory/compute constraints
- ❌ Neglect logging and visualization infrastructure
- ❌ Skip baseline comparisons

## Planning Output Format

Structure your plan as follows:

```markdown
# Plan: [Feature/Project Name]

## Goal
[Clear statement of what we're building and why]

## Success Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Architecture Overview
[High-level design with component diagram if complex]

## Implementation Steps

### Step 1: [Component/Phase Name]
**Files to Create/Modify:**
- `path/to/file1.py` - Description
- `path/to/file2.py` - Description

**Tasks:**
1. Task description
2. Task description

**Testing:**
- How to validate this step

**Estimated Complexity:** Low/Medium/High

---

### Step 2: [Next Component/Phase]
[Repeat structure]

---

## Dependencies
- Library 1 (version)
- Library 2 (version)

## Risks & Mitigations
1. **Risk:** Description
   **Mitigation:** Strategy

## Open Questions
1. Question needing decision?
2. Another question?

## Resources
- Paper/documentation links
- Reference implementations
- Tutorials
```

## Integration with Development Workflow

After planning:
1. **Review with user** - Confirm approach before implementation
2. **Update CONTINUITY.md** - Document decisions and plan summary
3. **Create GitHub issue/project** - Track implementation progress
4. **Hand off to coder skill** - Detailed plan enables focused implementation

## Example Planning Scenarios

### Scenario 1: "Implement PPO algorithm"
**Plan would include:**
- Research phase: PPO paper, existing implementations
- Architecture: Actor-critic networks, advantage estimation
- Components: Policy network, value network, rollout buffer, loss functions
- Steps: Network implementation → Buffer → Training loop → Testing
- Validation: CartPole benchmark, compare to Stable-Baselines3

### Scenario 2: "Build custom GridWorld environment"
**Plan would include:**
- Specification: Grid size, observation (image/vector), actions, rewards
- Implementation: State representation, transition dynamics, rendering
- Validation: Space checks, episode tests, API compliance
- Integration: Registration, wrappers, example usage

### Scenario 3: "Add distributed training support"
**Plan would include:**
- Architecture decision: Ray vs custom multiprocessing
- Components: Worker processes, parameter server, aggregation
- Steps: Refactor training loop → Add distribution layer → Testing
- Performance: Scaling tests, overhead measurement

---

**Ready to plan!** When invoked, I'll thoroughly analyze your request, explore the codebase, research best practices, and provide a detailed, actionable implementation plan.
