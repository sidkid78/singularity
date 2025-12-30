# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Agentic Layer Framework** - a Python framework for building multi-agent AI systems using the Google GenAI SDK (Gemini models). The framework implements a Class/Grade progression system for measuring and improving agentic capability.

## Commands

### Installation & Setup

```bash
cd backend
pip install -e .                    # Install package in editable mode
pip install -e ".[dev]"             # Install with dev dependencies
export GEMINI_API_KEY="your-key"    # Required for agent execution
```

### Running

```bash
python backend/demo.py              # Run demo script
python backend/main.py              # Run main entry point
```

### Testing & Linting

```bash
pytest backend/tests                # Run all tests
pytest backend/tests -k "test_name" # Run specific test
black backend/                      # Format code
ruff check backend/                 # Lint code
ruff check backend/ --fix           # Auto-fix lint issues
```

## Architecture

### Core Concepts

**Classification System:**

- **Classes** (1-3): Scale of operation (single codebase → multi-codebase → autonomous)
- **Grades** (1-5): Capability level within a class (basic prompts → full orchestration)

**Execution Pattern:** Request → Validate → Resolve (closed-loop self-correction)

### Module Structure (`backend/agentic_layer/`)

| Module | Purpose |
|--------|---------|
| `layer.py` | `AgenticLayer` - main entry point. Wraps codebase, manages agents/tools/workflows |
| `orchestrator.py` | `OrchestratorAgent` - Grade 5 "lead agent" that delegates to specialized agents |
| `agents.py` | `BaseAgent`, `ClosedLoopAgent`, and factory functions for specialized agents (coder, reviewer, tester, planner) |
| `tools_manager.py` | `AgentToolsManager` - curates tools per task to prevent "Grade 3 trap" (tool overload) |
| `agent_types.py` | Pydantic models, enums (`AgentRole`, `AgenticClass`, `AgenticGrade`), task/result schemas |
| `core.py` | Re-exports all public API |

### Key Patterns

**Tool Registration:**

```python
layer.register_tool(name, description, parameters, implementation)
layer.register_function(python_callable)  # Auto-generates schema from type hints
layer.register_skill(name, tools, instructions)  # Combines tools for specific tasks
```

**AI Developer Workflows (ADWs):** Built-in workflow definitions in `orchestrator.py`:

- `plan_build`, `plan_build_review`, `plan_build_review_fix`, `test_driven`
- Custom workflows via `WorkflowDefinition` Pydantic model

**Agent Roles:** `ORCHESTRATOR`, `PLANNER`, `CODER`, `REVIEWER`, `TESTER`, `DEBUGGER`, `DOCUMENTER`, `RESEARCHER`

### State & Persistence

- State saved to `.agentic_layer/state.json`
- Trust metrics tracked via `TrustMetrics` model (success rate, tool call chains, human interventions)
- Codebase context auto-loaded from `README.md`, `CLAUDE.md`, `.claude/claude.md`, `docs/ARCHITECTURE.md`

## Dependencies

- **google-genai** (>=1.56.0): Core LLM SDK for Gemini models
- **pydantic**: All data models use Pydantic v2 for validation
- Python 3.12+ required
