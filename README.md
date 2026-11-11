[![progress-banner](https://backend.codecrafters.io/progress/claude-code/af2a7796-81c2-4d0e-851f-66522bb59ee5)](https://app.codecrafters.io/users/chopdekaradhrit2013-droid?r=2qF)

# Build Your Own Claude Code (Python)

A fully-featured autonomous AI coding assistant implemented from scratch in Python, completing all stages and extensions of the [CodeCrafters "Build Your Own Claude Code" challenge](https://codecrafters.io/challenges/claude-code).

---

## Features

### 1. Autonomous Agent Loop & Core Tools
- **Agent Loop**: Dynamic conversation loop executing multi-turn tool calling and reasoning with LLMs.
- **Read Tool**: File inspection with UTF-8 decoding.
- **Write Tool**: File modification and creation with automatic directory path generation.
- **Bash Tool**: Shell command execution capturing both `stdout` and `stderr`.

### 2. Claude Skills Extension
- **Skill Discovery**: Scans `.claude/skills/` for `SKILL.md` frontmatter and registers skills in system prompt catalogs.
- **Parameter Substitution**: Substitutes `$ARGUMENTS`, indexed arguments (`$ARGUMENTS[i]`), and shorthand placeholders (`$0`, `$1`, ...).
- **Skill Stacking**: Supports invoking multiple skills sequentially (`/skill1 /skill2 arg`).
- **Bundled Executable Scripts**: Resolves and executes bundled scripts in `scripts/` with relative folder contexts.
- **LLM-Invoked Skills**: Advertises the `Skill` tool for autonomous model selection.
- **Subagents (`context: fork`)**: Spawns isolated agent conversation stacks that execute skill instructions and return final answers to the parent thread.

### 3. Language Server Protocol (LSP) Extension
- **Client Protocol**: Connects to language servers via [`LSPClient`](file:///Users/manishachopdekar/adhrit/notyouraverageai/codecrafters-claude-code-python/app/lsp_client.py).
- **Definition Lookup (`goToDefinition`)**: Automatically opens documents (`textDocument/didOpen`), queries definitions (`textDocument/definition`), and returns relative file paths with 1-based coordinates.
- **Reference Resolution (`findReferences`)**: Finds all call sites and definitions (`textDocument/references`) across the project.
- **Incremental Synchronization**: Keeps the server in sync with disk changes via `textDocument/didChange` and document version tracking.
- **Diagnostics Reporting (`textDocument/publishDiagnostics`)**:
  - Asynchronously polls server notifications with cursor tracking.
  - Filters out stale reports by checking document versions while accepting unversioned reports.
  - Injects formatted `<new-diagnostics>` into tool results to guide the model through multi-file migrations.

---

## Running Locally

### Prerequisites
- Python 3.11+
- `uv` (optional, for fast virtualenv and dependency management)
- `OPENROUTER_API_KEY` (and optionally `OPENROUTER_BASE_URL`)

### Running the Agent

```bash
# Direct prompt invocation
./your_program.sh -p "What is 2 + 2?"

# Using a skill
./your_program.sh -p "/deploy staging eu-west"

# Running a refactoring or code inspection task
./your_program.sh -p "Use LSP goToDefinition on checkout.py at line 5, character 12"
```

### Running Tests

```bash
# Run tests with CodeCrafters CLI
codecrafters test

# Run tests across all stages
codecrafters test --previous
```
