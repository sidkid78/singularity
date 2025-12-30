"""
Agentic Layer Framework - Core Types
=====================================
Implements the Class/Grade classification system for agentic layers.

Classes:
- Class 1: Single codebase, scaling grades 1-5
- Class 2: Multi-codebase orchestration  
- Class 3: Full autonomous operation (codebase singularity)

Grades (within each class):
- Grade 1: Prime prompt + memory files (minimal)
- Grade 2: Specialized prompts + sub-agents + planning
- Grade 3: Skills + MCP + Custom Tools
- Grade 4: Closed-loop prompts + feedback (self-correcting)
- Grade 5: AI Developer Workflows + full orchestration
"""

from enum import Enum
from typing import Any, Callable, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================
# ENUMS
# ============================================

class AgentRole(str, Enum):
    """Specialized agent roles for task delegation"""
    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    CODER = "coder"
    REVIEWER = "reviewer"
    TESTER = "tester"
    DOCUMENTER = "documenter"
    DEBUGGER = "debugger"
    RESEARCHER = "researcher"


class AgenticClass(int, Enum):
    """Agentic layer classification"""
    CLASS_1 = 1  # Single codebase
    CLASS_2 = 2  # Multi-codebase
    CLASS_3 = 3  # Autonomous (codebase singularity)


class AgenticGrade(int, Enum):
    """Agentic layer grade within a class"""
    GRADE_1 = 1  # Prime prompt + memory
    GRADE_2 = 2  # Sub-agents + planning
    GRADE_3 = 3  # Custom tools + skills
    GRADE_4 = 4  # Closed-loop feedback
    GRADE_5 = 5  # Full orchestration + ADWs


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    SUCCESS = "success"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"
    CANCELLED = "cancelled"


class FeedbackLoopPhase(str, Enum):
    """Phases of the closed-loop execution pattern"""
    REQUEST = "request"    # Initial execution
    VALIDATE = "validate"  # Check results
    RESOLVE = "resolve"    # Fix issues or confirm


# ============================================
# TOOL DEFINITIONS
# ============================================

class ToolParameter(BaseModel):
    """Schema for a tool parameter"""
    name: str
    type: str  # string, integer, boolean, array, object
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[list[str]] = None


class ToolDefinition(BaseModel):
    """Definition of a tool available to agents"""
    name: str
    description: str
    parameters: list[ToolParameter] = []
    category: str = "general"  # general, database, filesystem, api, etc.
    token_cost: int = 0  # Estimated tokens for tool schema
    requires_confirmation: bool = False  # High-risk tools
    
    def to_function_declaration(self) -> dict:
        """Convert to Gemini function declaration format"""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {"type": param.type.upper(), "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "OBJECT",
                "properties": properties,
                "required": required
            }
        }


class Skill(BaseModel):
    """A skill teaches an agent how to use specific tools for a task"""
    name: str
    description: str
    tools_required: list[str]  # Tool names this skill uses
    system_prompt: str  # Instructions for using this skill
    example_usage: Optional[str] = None
    

# ============================================
# TASK & EXECUTION MODELS
# ============================================

class TaskDefinition(BaseModel):
    """Definition of a task to be executed"""
    task_id: str
    description: str
    assigned_role: AgentRole
    priority: int = Field(default=1, ge=1, le=10)
    dependencies: list[str] = []  # Other task_ids that must complete first
    tools_allowed: list[str] = []  # Specific tools for this task
    max_iterations: int = 5  # For closed-loop execution
    timeout_seconds: int = 300
    created_at: datetime = Field(default_factory=datetime.now)


class TaskResult(BaseModel):
    """Result of task execution"""
    task_id: str
    status: TaskStatus
    agent_role: AgentRole
    output: str
    artifacts: list[str] = []  # File paths or references created
    tool_calls_made: int = 0
    iterations_used: int = 1
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    needs_human_review: bool = False
    feedback_notes: list[str] = []


class ExecutionPlan(BaseModel):
    """Plan created by orchestrator for task execution"""
    plan_id: str
    goal: str
    tasks: list[TaskDefinition]
    execution_order: list[str]  # Task IDs in order
    parallel_groups: list[list[str]] = []  # Groups that can run in parallel
    estimated_duration_seconds: int = 0
    created_at: datetime = Field(default_factory=datetime.now)


# ============================================
# FEEDBACK & VALIDATION
# ============================================

class ValidationResult(BaseModel):
    """Result of validating task output"""
    is_valid: bool
    issues: list[str] = []
    suggestions: list[str] = []
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    requires_retry: bool = False
    

class FeedbackLoop(BaseModel):
    """Tracks a closed-loop execution cycle"""
    loop_id: str
    task_id: str
    current_phase: FeedbackLoopPhase
    iteration: int = 1
    max_iterations: int = 5
    history: list[dict] = []  # Track each iteration's results
    
    def record_iteration(self, phase: FeedbackLoopPhase, result: dict):
        self.history.append({
            "iteration": self.iteration,
            "phase": phase.value,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        

# ============================================
# AGENT CONFIGURATION
# ============================================

class AgentConfig(BaseModel):
    """Configuration for an agent instance"""
    role: AgentRole
    model: str = "gemini-2.5-flash"
    system_instruction: str
    tools: list[str] = []  # Tool names available to this agent
    skills: list[str] = []  # Skill names this agent can use
    thinking_budget: int = 0  # 0 = off, 128+ for pro
    temperature: float = 0.7
    max_output_tokens: Optional[int] = None
    

class OrchestratorConfig(AgentConfig):
    """Extended config for orchestrator agent"""
    role: AgentRole = AgentRole.ORCHESTRATOR
    model: str = "gemini-2.5-pro"  # Pro for complex planning
    thinking_budget: int = 1024
    can_spawn_agents: bool = True
    can_run_workflows: bool = True
    max_concurrent_agents: int = 10


# ============================================
# METRICS & TRUST
# ============================================

class TrustMetrics(BaseModel):
    """Metrics for measuring agent trust (private benchmarks)"""
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_tool_calls: int = 0
    successful_tool_chains: int = 0
    longest_successful_chain: int = 0
    avg_chain_length: float = 0.0
    avg_iterations_to_success: float = 0.0
    human_interventions: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks
    
    @property
    def trust_score(self) -> float:
        """Calculate overall trust score (0-100)"""
        if self.total_tasks == 0:
            return 0.0
        
        # Weighted factors
        success_weight = 0.5
        chain_weight = 0.3
        intervention_weight = 0.2
        
        success_factor = self.success_rate
        chain_factor = min(self.avg_chain_length / 10, 1.0)  # Normalize to 10 calls
        intervention_factor = 1.0 - min(self.human_interventions / self.total_tasks, 1.0)
        
        score = (
            success_factor * success_weight +
            chain_factor * chain_weight +
            intervention_factor * intervention_weight
        ) * 100
        
        return round(score, 2)


# ============================================
# AGENTIC LAYER STATE
# ============================================

class AgenticLayerState(BaseModel):
    """Current state of the agentic layer"""
    agentic_class: AgenticClass = AgenticClass.CLASS_1
    grade: AgenticGrade = AgenticGrade.GRADE_1
    active_agents: int = 0
    pending_tasks: int = 0
    completed_tasks: int = 0
    trust_metrics: TrustMetrics = Field(default_factory=TrustMetrics)
    last_activity: Optional[datetime] = None
    
    def upgrade_grade(self) -> bool:
        """Attempt to upgrade to next grade based on metrics"""
        if self.grade.value < 5:
            # Could add validation logic here
            self.grade = AgenticGrade(self.grade.value + 1)
            return True
        return False