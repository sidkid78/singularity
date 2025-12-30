"""
Agentic Layer Framework - Core Module
======================================
"""

from .agent_types import (
    # Enums
    AgentRole,
    AgenticClass,
    AgenticGrade,
    TaskStatus,
    FeedbackLoopPhase,
    
    # Tool definitions
    ToolDefinition,
    ToolParameter,
    Skill,
    
    # Task models
    TaskDefinition,
    TaskResult,
    ExecutionPlan,
    
    # Feedback models
    ValidationResult,
    FeedbackLoop,
    
    # Agent config
    AgentConfig,
    OrchestratorConfig,
    
    # Metrics
    TrustMetrics,
    AgenticLayerState,
)

from .tools_manager import (
    AgentToolsManager,
    create_tool,
    create_skill,
)

from .agents import (
    BaseAgent,
    ClosedLoopAgent,
    create_coder_agent,
    create_reviewer_agent,
    create_tester_agent,
    create_planner_agent,
)

from .orchestrator import (
    OrchestratorAgent,
    WorkflowDefinition,
    create_orchestrator,
    BUILT_IN_WORKFLOWS,
)

from .layer import (
    AgenticLayer,
    create_agentic_layer,
)

__all__ = [
    # Enums
    "AgentRole",
    "AgenticClass", 
    "AgenticGrade",
    "TaskStatus",
    "FeedbackLoopPhase",
    
    # Types
    "ToolDefinition",
    "ToolParameter",
    "Skill",
    "TaskDefinition",
    "TaskResult",
    "ExecutionPlan",
    "ValidationResult",
    "FeedbackLoop",
    "AgentConfig",
    "OrchestratorConfig",
    "TrustMetrics",
    "AgenticLayerState",
    
    # Tools
    "AgentToolsManager",
    "create_tool",
    "create_skill",
    
    # Agents
    "BaseAgent",
    "ClosedLoopAgent",
    "create_coder_agent",
    "create_reviewer_agent",
    "create_tester_agent",
    "create_planner_agent",
    
    # Orchestration
    "OrchestratorAgent",
    "WorkflowDefinition",
    "create_orchestrator",
    "BUILT_IN_WORKFLOWS",
    
    # Main interface
    "AgenticLayer",
    "create_agentic_layer",
]