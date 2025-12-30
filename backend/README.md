# Agentic Layer Framework

A production-ready Python framework for building **agentic layers** around your codebase using the Google GenAI SDK.

Based on the "Year of Trust" and "Agentic Layer" engineering frameworks, this library provides:

- 🎯 **Class/Grade progression system** for measuring agentic capability
- 🤖 **Multi-agent orchestration** with specialized agents
- 🔄 **Closed-loop self-correcting execution** (Request → Validate → Resolve)
- 📊 **Trust measurement** through tool calling metrics
- 🚀 **AI Developer Workflows (ADWs)** for common patterns

## The Agentic Layer Concept

The **agentic layer** is the new ring around your codebase where you teach your agents to operate your application on your behalf—as well or better than you ever could.

```
┌─────────────────────────────────────┐
│         AGENTIC LAYER               │
│  ┌─────────────────────────────┐    │
│  │    APPLICATION LAYER        │    │
│  │  ┌─────────────────────┐    │    │
│  │  │   Your Code         │    │    │
│  │  │   Database          │    │    │
│  │  │   Frontend/Backend  │    │    │
│  │  └─────────────────────┘    │    │
│  └─────────────────────────────┘    │
│                                     │
│  • Orchestrator Agent               │
│  • Specialized Agents               │
│  • Custom Tools & Skills            │
│  • Closed-Loop Execution            │
│  • AI Developer Workflows           │
└─────────────────────────────────────┘
```

## Classification System

### Classes

| Class | Description | Capability |
|-------|-------------|------------|
| **Class 1** | Single codebase | Basic to advanced agentic layer |
| **Class 2** | Multi-codebase | Orchestration across repositories |
| **Class 3** | Autonomous | "Codebase Singularity" - agents run better than humans |

### Grades (within Class 1)

| Grade | Key Elements | Compute Advantage |
|-------|--------------|-------------------|
| **Grade 1** | Prime prompt + memory files | Clean minimal setup |
| **Grade 2** | Sub-agents + planning | Parallelization |
| **Grade 3** | Skills + MCP + Custom Tools | Tool-enhanced agents |
| **Grade 4** | Closed-loop feedback | Self-correcting agents |
| **Grade 5** | Full orchestration + ADWs | Lead agent conducts team |

## Installation

```bash
pip install google-genai pydantic

# Or install this package
pip install -e .
```

Set your API key:

```bash
export GEMINI_API_KEY="your-api-key"
```

## Quick Start

```python
import asyncio
from agentic_layer import AgenticLayer

async def main():
    # Create your agentic layer
    layer = AgenticLayer(codebase_path="/path/to/your/project")
    
    # Simple execution with a coder agent
    result = await layer.execute(
        task="Fix the authentication bug in auth.py",
        role="coder"
    )
    print(result.output)
    
    # Full multi-agent orchestration
    result = await layer.orchestrate(
        "Build a REST API for user management with tests"
    )
    print(result["synthesis"])
    
    # Run a predefined workflow
    result = await layer.run_workflow(
        "plan_build_review",
        "Add caching to the database layer"
    )
    print(f"Success: {result['success']}")
    
    # Check your trust score
    print(f"Trust Score: {layer.get_trust_score()}/100")

asyncio.run(main())
```

## Core Components

### AgenticLayer

The main interface for your agentic layer:

```python
from agentic_layer import AgenticLayer

layer = AgenticLayer(
    codebase_path="/path/to/project",
    max_concurrent_agents=5,
    default_model="gemini-2.5-flash"
)

# Execute single tasks
result = await layer.execute("Write unit tests", role="tester")

# Full orchestration
result = await layer.orchestrate("Build feature X")

# Run workflows
result = await layer.run_workflow("plan_build_review", "Add logging")

# Quick helpers
result = await layer.quick_fix("Fix the null pointer in utils.py")
result = await layer.review("Review the auth module")
result = await layer.plan("Plan the migration to v2")
```

### Custom Tools

Register tools to extend agent capabilities:

```python
# Method 1: Manual definition
layer.register_tool(
    name="query_database",
    description="Execute a SQL query against the database",
    parameters={
        "query": ("string", "SQL query to execute", True),
        "database": ("string", "Database name", False)
    }
)

# Method 2: Register Python function directly
def analyze_code(file_path: str) -> dict:
    """Analyze code quality metrics for a file"""
    # Your implementation
    return {"complexity": 5, "issues": []}

layer.register_function(analyze_code)

# Method 3: Register a skill (combines tools)
layer.register_skill(
    name="refactor_module",
    description="Safely refactor a module",
    tools=["read_file", "write_file", "run_tests"],
    instructions="Always run tests after changes..."
)
```

### AI Developer Workflows

Built-in workflows:

| Workflow | Steps | Use Case |
|----------|-------|----------|
| `plan_build` | Plan → Build | Quick implementation |
| `plan_build_review` | Plan → Build → Review | Standard development |
| `plan_build_review_fix` | Plan → Build → Review → Fix | Full cycle |
| `test_driven` | Plan → Tests → Build → Test → Fix | TDD approach |

Register custom workflows:

```python
from agentic_layer import WorkflowDefinition, AgentRole

layer.register_workflow(WorkflowDefinition(
    name="security_audit",
    description="Security-focused development",
    steps=["plan", "build", "security_review", "fix", "test"],
    agent_roles=[
        AgentRole.PLANNER,
        AgentRole.CODER,
        AgentRole.REVIEWER,
        AgentRole.CODER,
        AgentRole.TESTER
    ]
))

result = await layer.run_workflow("security_audit", "Add payment processing")
```

### Trust Measurement

Track and improve agent reliability:

```python
# Get current metrics
metrics = layer.get_metrics()

print(f"Trust Score: {metrics['trust']['score']}/100")
print(f"Success Rate: {metrics['trust']['success_rate']*100}%")
print(f"Tool Calls: {metrics['trust']['tool_calls']}")
print(f"Longest Chain: {metrics['trust']['longest_chain']}")

# Assess your grade
assessment = layer.assess_grade()
print(f"Current Grade: {assessment['current_grade']}")
print(f"Recommendations: {assessment['recommendations']}")
```

### Closed-Loop Execution

Self-correcting agents that validate their own work:

```python
# Enable closed-loop execution (default for most tasks)
result = await layer.execute(
    task="Implement OAuth2 authentication",
    role="coder",
    use_closed_loop=True  # Request → Validate → Resolve cycle
)

print(f"Iterations used: {result.iterations_used}")
print(f"Confidence: {result.confidence_score}")
print(f"Feedback: {result.feedback_notes}")
```

## Architecture

```
agentic_layer/
├── __init__.py          # Main exports
├── core/
│   ├── agent_types.py         # Pydantic models, enums, schemas
│   ├── tools_manager.py # Tool registration and management
│   ├── agents.py        # Base and specialized agents
│   ├── orchestrator.py  # Lead agent orchestration
│   └── layer.py         # Main AgenticLayer interface
├── demo.py              # Usage examples
└── pyproject.toml       # Package config
```

## Agent Roles

| Role | Responsibility | Model | Thinking |
|------|---------------|-------|----------|
| **ORCHESTRATOR** | Plan, delegate, synthesize | gemini-2.5-pro | High (1024) |
| **PLANNER** | Architecture, design | gemini-2.5-pro | High (1024) |
| **CODER** | Implementation | gemini-2.5-flash | Off (fast) |
| **REVIEWER** | Quality, security | gemini-2.5-flash | Medium (256) |
| **TESTER** | Tests, validation | gemini-2.5-flash | Off (fast) |
| **DEBUGGER** | Bug investigation | gemini-2.5-flash | Medium |

## Best Practices

### Grade 3 Trap (Tool Overload)

Don't overwhelm agents with tools:

```python
# ❌ Bad: Too many tools
layer.tools_manager.max_tools_per_request = 50  # Agent gets confused

# ✅ Good: Curated tools per task
layer.tools_manager.max_tools_per_request = 10
tools = layer.tools_manager.get_tools_for_task(
    task_description="Fix the bug",
    role=AgentRole.CODER,
    required_tools=["read_file", "write_file"]
)
```

### Closed-Loop for Critical Tasks

Always use closed-loop for important work:

```python
# For critical tasks, increase iterations
from agentic_layer.core.agent_types import TaskDefinition

task = TaskDefinition(
    task_id="critical_1",
    description="Migrate database schema",
    assigned_role=AgentRole.CODER,
    max_iterations=10  # More chances to self-correct
)
```

### Save State for Persistence

```python
# Save after significant work
layer.save_state()  # Saves to .agentic_layer/state.json

# Load on startup
layer.load_state()
```

## Roadmap

- [ ] Class 2: Multi-codebase support
- [ ] MCP server integration
- [ ] Async task queue (out-of-loop)
- [ ] Web UI for monitoring
- [ ] Private benchmark suite
- [ ] Agent sandboxing

## Contributing

Contributions welcome! Focus areas:

1. Additional agent specializations
2. More built-in workflows
3. Better tool schemas
4. Trust measurement improvements

## License

MIT

---

*"The agentic layer is the new ring around your codebase where you teach your agents to operate your application on your behalf."*
