"""
Agentic Layer Framework
========================
A production-ready framework for building agentic layers around your codebase.

Based on the "Year of Trust" and "Agentic Layer" frameworks:
- Class/Grade progression system
- Multi-agent orchestration
- Closed-loop self-correcting execution
- Trust measurement through tool calling metrics
- AI Developer Workflows (ADWs)

Quick Start:
    from agentic_layer import AgenticLayer
    
    # Create your agentic layer
    layer = AgenticLayer(codebase_path="/path/to/project")
    
    # Simple execution
    result = await layer.execute("Fix the auth bug")
    
    # Full orchestration
    result = await layer.orchestrate("Build user management API")
    
    # Run a workflow
    result = await layer.run_workflow("plan_build_review", "Add caching")

Requirements:
    pip install google-genai pydantic
    
    Set GEMINI_API_KEY environment variable
"""

from .core import (
    # Main interface
    AgenticLayer,
    create_agentic_layer,
    
    # Orchestration
    OrchestratorAgent,
    create_orchestrator,
    WorkflowDefinition,
    BUILT_IN_WORKFLOWS,
    
    # Agents
    BaseAgent,
    ClosedLoopAgent,
    create_coder_agent,
    create_reviewer_agent,
    create_tester_agent,
    create_planner_agent,
    
    # Tools
    AgentToolsManager,
    create_tool,
    create_skill,
    
    # Types
    AgentRole,
    AgenticClass,
    AgenticGrade,
    TaskStatus,
    TaskDefinition,
    TaskResult,
    TrustMetrics,
)

__version__ = "0.1.0"
__author__ = "Chris"

__all__ = [
    # Main
    "AgenticLayer",
    "create_agentic_layer",
    
    # Orchestration
    "OrchestratorAgent",
    "create_orchestrator",
    "WorkflowDefinition",
    "BUILT_IN_WORKFLOWS",
    
    # Agents
    "BaseAgent",
    "ClosedLoopAgent",
    "create_coder_agent",
    "create_reviewer_agent",
    "create_tester_agent",
    "create_planner_agent",
    
    # Tools
    "AgentToolsManager",
    "create_tool",
    "create_skill",
    
    # Types
    "AgentRole",
    "AgenticClass",
    "AgenticGrade",
    "TaskStatus",
    "TaskDefinition",
    "TaskResult",
    "TrustMetrics",
]