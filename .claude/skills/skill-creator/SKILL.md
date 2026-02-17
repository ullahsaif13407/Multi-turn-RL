---
name: skill-creator
description: Create new Claude Code skills following the official format. Use when the user asks to create a new skill, slash command, or wants to extend Claude's capabilities with custom instructions.
disable-model-invocation: false
---

# Skill Creator

Create properly formatted skills for Claude Code following the official Agent Skills standard.

## What You Need to Know About Skills

Skills extend Claude's capabilities by providing specialized instructions. They can be:
- **Automatically invoked** by Claude when relevant to the conversation
- **Manually invoked** by users with `/skill-name`
- **Reference knowledge** (coding standards, patterns, domain knowledge)
- **Task workflows** (deployments, commits, testing procedures)

## Skill Structure

Every skill requires a directory with a `SKILL.md` file:

```
.claude/skills/my-skill/
├── SKILL.md           # Main instructions (required)
├── template.md        # Optional: Template for Claude to fill
├── examples/          # Optional: Example outputs
│   └── sample.md
└── scripts/           # Optional: Helper scripts
    └── helper.py
```

## SKILL.md Format

Every `SKILL.md` must have:

1. **YAML frontmatter** (between `---` markers)
2. **Markdown instructions** (what Claude should do)

### Template

```yaml
---
name: skill-name
description: Clear description of what this skill does and when to use it. Include keywords users would naturally say.
---

# Skill Name

[Your instructions here that Claude will follow when this skill is active]

## When to Use
- Situation 1
- Situation 2

## Instructions
1. Step 1
2. Step 2

## Examples
- Example usage 1
- Example usage 2

## Guidelines
- Guideline 1
- Guideline 2
```

## Frontmatter Fields

All fields are optional except `description` (recommended):

| Field | Purpose | Example |
|-------|---------|---------|
| `name` | Skill name (becomes `/name`). Lowercase, hyphens, max 64 chars. | `my-skill` |
| `description` | What it does and when to use it. Claude uses this to decide when to load it. | `Explains code with diagrams` |
| `argument-hint` | Hint for autocomplete | `[filename] [format]` |
| `disable-model-invocation` | Set `true` to prevent Claude from auto-loading (manual `/name` only) | `true` |
| `user-invocable` | Set `false` to hide from `/` menu (background knowledge) | `false` |
| `allowed-tools` | Tools Claude can use without asking | `Read, Grep, Glob` |
| `model` | Model to use | `haiku` |
| `context` | Set `fork` to run in isolated subagent | `fork` |
| `agent` | Subagent type when `context: fork` | `Explore` |

## Common Skill Patterns

### Pattern 1: Reference Knowledge (Auto-invoked)

```yaml
---
name: api-conventions
description: API design patterns and conventions for this codebase
---

When writing API endpoints:
- Use RESTful naming conventions
- Return consistent error formats
- Include request validation

BAD:
```js
app.get('/getData', ...)
```

GOOD:
```js
app.get('/api/v1/data', ...)
```
```

### Pattern 2: Task Workflow (Manual invocation)

```yaml
---
name: deploy
description: Deploy the application to production
disable-model-invocation: true
---

Deploy the application to production:

1. Run the test suite
2. Build the application
3. Push to the deployment target
4. Verify the deployment succeeded

IMPORTANT: Always ask for confirmation before pushing to production.
```

### Pattern 3: With Arguments

```yaml
---
name: fix-issue
description: Fix a GitHub issue
argument-hint: [issue-number]
disable-model-invocation: true
---

Fix GitHub issue $ARGUMENTS following our coding standards.

1. Read the issue description using `gh issue view $ARGUMENTS`
2. Understand the requirements
3. Implement the fix
4. Write tests
5. Create a commit with message: "Fix #$ARGUMENTS"
```

### Pattern 4: Research with Subagent

```yaml
---
name: deep-research
description: Research a topic thoroughly in the codebase
context: fork
agent: Explore
---

Research $ARGUMENTS thoroughly:

1. Find relevant files using Glob and Grep
2. Read and analyze the code
3. Summarize findings with specific file references
4. Identify patterns and relationships
```

### Pattern 5: Dynamic Context Injection

```yaml
---
name: pr-summary
description: Summarize changes in a pull request
context: fork
agent: Explore
allowed-tools: Bash(gh:*)
---

## Pull request context
- PR diff: !`gh pr diff`
- PR comments: !`gh pr view --comments`
- Changed files: !`gh pr diff --name-only`

## Your task
Summarize this pull request:
1. What changed and why
2. Potential impacts
3. Review recommendations
```

## Advanced Features

### Enable Extended Thinking

Include the word **"ultrathink"** anywhere in your skill content to enable extended thinking mode.

```yaml
---
name: complex-analysis
description: Deep analysis requiring extended thinking
---

Perform ultrathink analysis of $ARGUMENTS...
```

### Dynamic Commands with !`command`

The `!`command`` syntax runs shell commands before the skill content is sent to Claude:

```yaml
---
name: git-context
description: Provide git context for current work
---

Current git status:
!`git status --short`

Recent commits:
!`git log --oneline -5`

Based on this context, help with $ARGUMENTS.
```

### String Substitutions

Available variables:
- `$ARGUMENTS` - All arguments passed when invoking
- `${CLAUDE_SESSION_ID}` - Current session ID

## Skill Locations

| Location | Path | Applies to |
|----------|------|-----------|
| Personal | `~/.claude/skills/<name>/SKILL.md` | All your projects |
| Project | `.claude/skills/<name>/SKILL.md` | This project only |
| Plugin | `<plugin>/skills/<name>/SKILL.md` | Where plugin is enabled |

## Best Practices

### DO:
- ✅ Write clear, specific descriptions with natural keywords
- ✅ Include concrete examples (BAD vs GOOD)
- ✅ Keep SKILL.md under 500 lines (use supporting files for details)
- ✅ Use `disable-model-invocation: true` for workflows with side effects
- ✅ Use descriptive names with hyphens: `fix-issue`, `api-conventions`
- ✅ Structure with clear hierarchy: overview, prerequisites, steps, examples
- ✅ Use bullet points and code blocks for scannability

### DON'T:
- ❌ Make descriptions too generic ("helps with code")
- ❌ Include sensitive data in skills (use env vars instead)
- ❌ Create huge monolithic skills (break into multiple)
- ❌ Use `context: fork` for pure reference content (no task)
- ❌ Forget to test both manual invocation and auto-loading

## Creating a New Skill - Workflow

When the user asks to create a skill, follow these steps:

1. **Ask clarifying questions**:
   - What should this skill do?
   - When should it be triggered? (keywords for auto-loading)
   - Should it be auto-invoked by Claude or manual only?
   - Does it need arguments?
   - Does it need special tools or permissions?

2. **Determine the skill type**:
   - Reference knowledge → auto-invoked, inline
   - Task workflow → manual invocation, `disable-model-invocation: true`
   - Research/analysis → `context: fork`, `agent: Explore`
   - Workflow with side effects → manual only, restricted tools

3. **Choose location**:
   - Project-specific → `.claude/skills/<name>/`
   - Personal (all projects) → `~/.claude/skills/<name>/`

4. **Create the directory**:
   ```bash
   mkdir -p .claude/skills/<name>
   ```

5. **Write SKILL.md** with:
   - Proper YAML frontmatter
   - Clear instructions
   - Examples
   - Guidelines

6. **Test the skill**:
   - Try auto-invocation (if enabled)
   - Try manual invocation: `/skill-name`
   - Verify it appears in available skills

## Example Session

**User**: "Create a skill that helps me write better commit messages"

**Your Response**:
```
I'll create a commit message helper skill. A few questions:

1. Should this run automatically when you type `/commit`, or should Claude suggest it?
2. Do you want it to analyze staged changes first?
3. Any specific commit message format you follow?

Based on typical needs, I'll create a skill that:
- Only runs when you invoke `/commit` (prevents auto-commits)
- Analyzes git diff
- Follows conventional commits format
- Asks for confirmation

[Create the skill]
```

## Testing Skills

After creating a skill:

1. **Check it's available**: Ask "What skills are available?"
2. **Test auto-invocation**: Say something matching the description
3. **Test manual invocation**: Type `/skill-name`
4. **Verify output**: Does it follow the instructions?
5. **Refine description**: Adjust keywords if it doesn't trigger when expected

## Troubleshooting

**Skill not triggering?**
- Check description includes natural keywords
- Verify it appears in available skills list
- Try manual invocation first: `/skill-name`
- Make description more specific

**Triggering too often?**
- Make description more specific
- Add `disable-model-invocation: true` for manual only

**Claude doesn't see all skills?**
- Skill descriptions exceed context budget (default 15,000 chars)
- Set `SLASH_COMMAND_TOOL_CHAR_BUDGET` environment variable
- Or use `user-invocable: false` for background skills

## Quick Reference

```yaml
# Minimal skill (auto-invoked)
---
name: my-skill
description: What it does and when to use it
---
Instructions here...
```

```yaml
# Manual only (no auto-invoke)
---
name: deploy
description: Deploy to production
disable-model-invocation: true
---
Steps here...
```

```yaml
# Background knowledge (no manual invoke)
---
name: legacy-context
description: Context about legacy system
user-invocable: false
---
Knowledge here...
```

```yaml
# Research task (forked subagent)
---
name: research
description: Deep research task
context: fork
agent: Explore
---
Research instructions...
```

```yaml
# With dynamic context
---
name: pr-helper
description: Help with pull requests
allowed-tools: Bash(gh:*)
---
PR status: !`gh pr status`
Instructions...
```

## Resources

- Official docs: https://code.claude.com/docs/en/skills
- Agent Skills standard: https://agentskills.io
- Example skills: https://github.com/anthropics/skills
- Community skills: https://github.com/travisvn/awesome-claude-skills

---

Ready to create skills! When the user asks to create a skill, gather requirements and generate a properly formatted SKILL.md following this guide.
