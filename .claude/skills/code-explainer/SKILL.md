---
name: code-explainer
description: Explain codebase architecture with ASCII diagrams. Use when asking to explain, understand, or visualize code structure, system design, data flow, or component relationships in a project.
argument-hint: [path]
---

# Code Explainer - Architecture & System Visualizer

Explain codebase structure and system design using clear ASCII diagrams and concise descriptions.

## When to Use This Skill

- Explaining overall codebase architecture
- Visualizing component relationships
- Understanding data flow through a system
- Documenting system design with diagrams
- Onboarding to a new codebase
- Creating technical documentation

## Analysis Process

1. **Scan Directory Structure** - Map files and folders
2. **Identify Key Components** - Entry points, core modules, configs
3. **Trace Dependencies** - Imports, function calls, data flow
4. **Generate ASCII Diagrams** - Visual representation
5. **Write Clear Explanations** - Concise summaries

## Output Format

### 1. Overview Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     PROJECT: <name>                          │
├─────────────────────────────────────────────────────────────┤
│  Purpose: <one-line description>                            │
│  Entry:   <main entry point(s)>                             │
│  Tech:    <key technologies/frameworks>                     │
└─────────────────────────────────────────────────────────────┘
```

### 2. Directory Tree (Annotated)

```
project/
├── src/               # Source code
│   ├── core/          # Core business logic
│   ├── api/           # External interfaces
│   └── utils/         # Shared utilities
├── configs/           # Configuration files
├── tests/             # Test suite
└── scripts/           # Entry points
```

### 3. Component Diagram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Module A  │────▶│   Module B  │────▶│   Module C  │
│  (purpose)  │     │  (purpose)  │     │  (purpose)  │
└─────────────┘     └──────┬──────┘     └─────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │   Module D  │
                   │  (purpose)  │
                   └─────────────┘
```

### 4. Data Flow Diagram

```
Input ──▶ Process ──▶ Transform ──▶ Output
  │          │            │           │
  ▼          ▼            ▼           ▼
[type]    [logic]      [logic]     [type]
```

### 5. Class/Module Relationships

```
┌──────────────────┐
│   BaseClass      │
│  ─────────────   │
│  + method1()     │
│  + method2()     │
└────────┬─────────┘
         │ extends
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌───────┐
│ Sub A │  │ Sub B │
└───────┘  └───────┘
```

## ASCII Diagram Toolkit

### Boxes and Containers

```
┌───────────────┐    ╔═══════════════╗    +---------------+
│  Single Line  │    ║  Double Line  ║    |  ASCII Only   |
└───────────────┘    ╚═══════════════╝    +---------------+
```

### Arrows and Connectors

```
Horizontal:  ───▶  ◀───  ◀──▶  ────
Vertical:    │     ▲     ▼
             │     │     │
             ▼     │     ▲

Corners:     ┌─    ─┐    └─    ─┘
             │      │    │      │

T-junctions: ├─    ─┤    ┬     ┴
             │      │    │     │

Cross:       ┼
             │
```

### Relationship Types

```
Dependency:     A ────▶ B       (A depends on B)
Bidirectional:  A ◀───▶ B       (mutual dependency)
Inheritance:    A ───|▶ B       (A extends B)
Composition:    A ◆────▶ B      (A contains B)
Association:    A ─────── B     (A relates to B)
Data Flow:      A ══════▶ B     (data flows A to B)
```

### Grouping

```
┌─────────── Module ───────────┐
│  ┌─────┐  ┌─────┐  ┌─────┐  │
│  │  A  │  │  B  │  │  C  │  │
│  └─────┘  └─────┘  └─────┘  │
└─────────────────────────────┘
```

### Layers

```
┌───────────────────────────────┐
│         Presentation          │
├───────────────────────────────┤
│          Business             │
├───────────────────────────────┤
│            Data               │
└───────────────────────────────┘
```

## Explanation Guidelines

### Be Concise

```
BAD:  "The DataLoader class is responsible for the loading of data 
       from various sources and it processes them..."

GOOD: "DataLoader: Loads and preprocesses training data from disk/remote."
```

### Use Tables for Reference

```
| Component     | File              | Purpose                    |
|---------------|-------------------|----------------------------|
| Entry Point   | train.py          | CLI & training orchestration|
| Policy        | policy.py         | Neural network forward pass |
| Algorithm     | grpo.py           | GRPO training logic         |
```

### Highlight Key Patterns

```
Design Patterns Used:
• Factory Pattern    → Environment creation
• Strategy Pattern   → Algorithm selection  
• Observer Pattern   → Logging callbacks
```

### Show Configuration Flow

```
config.yaml ──▶ ConfigLoader ──▶ Dataclass
                    │
                    ├──▶ Environment Config
                    ├──▶ Model Config
                    └──▶ Training Config
```

## Analysis Depth Levels

### Level 1: Bird's Eye (Default)
- Project purpose and scope
- Major components and their roles
- High-level data flow
- Entry points and outputs

### Level 2: Component Detail
- Individual module responsibilities
- Inter-module dependencies
- Key classes and functions
- Configuration options

### Level 3: Implementation Deep-Dive
- Algorithm specifics
- Data structures
- Performance considerations
- Extension points

## Example Output

```
┌─────────────────────────────────────────────────────────────┐
│                    tiny-RL Training System                   │
├─────────────────────────────────────────────────────────────┤
│  Purpose: Minimal GRPO training for LLM code generation     │
│  Entry:   train.py                                          │
│  Tech:    PyTorch, Transformers, Docker                     │
└─────────────────────────────────────────────────────────────┘

Directory Structure:
tiny-RL/
├── train.py        # Entry: CLI + training loop
├── grpo.py         # GRPO algorithm implementation
├── policy.py       # LLM policy wrapper
├── sandbox.py      # Docker-based code execution
├── data_loader.py  # Dataset loading
├── config.yaml     # Hyperparameters
└── tests/          # Test suite

Component Flow:
┌────────────┐    ┌────────────┐    ┌────────────┐
│ DataLoader │───▶│   Policy   │───▶│    GRPO    │
│ (prompts)  │    │   (LLM)    │    │ (training) │
└────────────┘    └─────┬──────┘    └─────┬──────┘
                        │                  │
                        ▼                  │
                 ┌────────────┐            │
                 │  Sandbox   │◀───────────┘
                 │ (execute)  │
                 └────────────┘

Data Flow:
prompt ──▶ policy.generate() ──▶ code ──▶ sandbox.run() ──▶ reward
                                                              │
                                                              ▼
                                          grpo.update() ◀─── score
```

## Workflow

1. **Receive path** (defaults to current directory if not specified)
2. **List directory** structure recursively
3. **Read key files**: entry points, configs, core modules
4. **Identify patterns**: imports, class hierarchies, data flow
5. **Generate diagrams** using ASCII toolkit
6. **Write explanations** at requested depth level
7. **Output formatted** markdown with diagrams

## Tips for Clear Diagrams

- Keep diagrams under 80 characters wide
- Align boxes and arrows consistently  
- Use whitespace for visual separation
- Label all arrows with relationship type
- Group related components in containers
- Show direction of data/control flow explicitly

---

**Ready to explain!** Provide a path or describe what you want explained, and receive clear ASCII diagrams with concise explanations of the codebase architecture.

