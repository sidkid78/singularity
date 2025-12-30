"""
Agentic Layer Manager
======================
The "ring around your codebase" that manages your agentic layer.

This is the main interface for:
- Initializing your agentic layer
- Tracking your class/grade progression
- Managing agents and tools
- Running orchestrated workflows
- Measuring trust through private benchmarks
"""

from google import genai
from typing import Optional, Callable
from pathlib import Path
from datetime import datetime
import json
import asyncio

from .agent_types import (
    AgenticClass,
    AgenticGrade,
    AgenticLayerState,
    TrustMetrics,
    TaskDefinition,
    TaskResult,
    Skill,
    ToolDefinition
)
from .tools_manager import AgentToolsManager, create_tool, create_skill
from .agents import (
    BaseAgent,
    ClosedLoopAgent,
    create_coder_agent,
    create_reviewer_agent,
    create_tester_agent,
    create_planner_agent
)
from .orchestrator import OrchestratorAgent, create_orchestrator, WorkflowDefinition


class AgenticLayer:
    """
    Main interface for your agentic layer.
    
    The agentic layer wraps around your codebase and provides:
    - Intelligent task execution with specialized agents
    - Multi-agent orchestration
    - Closed-loop self-correction
    - Trust measurement and optimization
    
    Usage:
        layer = AgenticLayer(codebase_path="/path/to/project")
        
        # Simple execution
        result = await layer.execute("Fix the bug in auth.py")
        
        # Orchestrated execution
        result = await layer.orchestrate("Build a REST API for users")
        
        # Run a workflow
        result = await layer.run_workflow("plan_build_review", "Add logging")
    """
    
    def __init__(
        self,
        codebase_path: Optional[str] = None,
        client: Optional[genai.Client] = None,
        max_concurrent_agents: int = 5,
        default_model: str = "gemini-2.5-flash"
    ):
        self.codebase_path = Path(codebase_path) if codebase_path else Path.cwd()
        self.client = client or genai.Client()
        self.default_model = default_model
        
        # Initialize components
        self.tools_manager = AgentToolsManager(max_tools_per_request=10)
        self.orchestrator = create_orchestrator(
            tools_manager=self.tools_manager,
            client=self.client,
            max_concurrent=max_concurrent_agents
        )
        
        # State tracking
        self.state = AgenticLayerState()
        self._initialized_at = datetime.now()
        self._execution_log: list[dict] = []
        
        # Load codebase context if available
        self._codebase_context = self._load_codebase_context()
    
    def _load_codebase_context(self) -> str:
        """Load context about the codebase from standard files"""
        context_parts = []
        
        # Check for common context files
        context_files = [
            "README.md",
            ".claude/claude.md",  # Memory file
            "CLAUDE.md",
            "docs/ARCHITECTURE.md",
            ".ai/context.md"
        ]
        
        for filename in context_files:
            filepath = self.codebase_path / filename
            if filepath.exists():
                try:
                    content = filepath.read_text()
                    context_parts.append(f"=== {filename} ===\n{content[:2000]}")
                except Exception:
                    pass
        
        return "\n\n".join(context_parts) if context_parts else ""
    
    # ============================================
    # GRADE ASSESSMENT
    # ============================================
    
    def assess_grade(self) -> dict:
        """Assess current agentic layer grade based on capabilities"""
        
        assessment = {
            "current_grade": self.state.grade.value,
            "capabilities": {},
            "recommendations": []
        }
        
        # Grade 1: Basic prompts + memory
        assessment["capabilities"]["has_memory"] = bool(self._codebase_context)
        
        # Grade 2: Sub-agents + planning
        assessment["capabilities"]["has_agents"] = len(self.orchestrator._agent_pool) > 0
        assessment["capabilities"]["has_planning"] = True  # Orchestrator always has this
        
        # Grade 3: Custom tools
        custom_tools = [t for t in self.tools_manager._tools.values() if t.category == "custom"]
        assessment["capabilities"]["has_custom_tools"] = len(custom_tools) > 0
        assessment["capabilities"]["custom_tool_count"] = len(custom_tools)
        
        # Grade 4: Closed-loop execution
        assessment["capabilities"]["has_closed_loop"] = True  # ClosedLoopAgent implemented
        
        # Grade 5: Full orchestration + ADWs
        assessment["capabilities"]["has_orchestration"] = True
        assessment["capabilities"]["workflow_count"] = len(self.orchestrator.workflows)
        
        # Determine grade
        if not assessment["capabilities"]["has_memory"]:
            suggested_grade = 1
            assessment["recommendations"].append(
                "Add a memory file (CLAUDE.md or .claude/claude.md) to reach Grade 1"
            )
        elif not assessment["capabilities"]["has_custom_tools"]:
            suggested_grade = 2
            assessment["recommendations"].append(
                "Register custom tools to reach Grade 3"
            )
        elif self.state.trust_metrics.total_tasks < 10:
            suggested_grade = 3
            assessment["recommendations"].append(
                "Run more tasks with closed-loop execution to validate Grade 4"
            )
        elif self.state.trust_metrics.trust_score < 70:
            suggested_grade = 4
            assessment["recommendations"].append(
                "Improve trust score above 70 to confidently claim Grade 5"
            )
        else:
            suggested_grade = 5
            assessment["recommendations"].append(
                "🎉 You've reached Grade 5! Consider advancing to Class 2 (multi-codebase)"
            )
        
        assessment["suggested_grade"] = suggested_grade
        
        return assessment
    
    # ============================================
    # TOOL & SKILL REGISTRATION
    # ============================================
    
    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict = {},
        implementation: Optional[Callable] = None,
        category: str = "custom"
    ):
        """Register a custom tool for agents to use"""
        
        tool = create_tool(
            name=name,
            description=description,
            parameters=parameters,
            category=category
        )
        
        self.tools_manager.register_tool(tool, implementation)
        print(f"✅ Registered tool: {name}")
    
    def register_function(self, func: Callable):
        """Register a Python function directly as a tool"""
        tool = self.tools_manager.register_python_function(func)
        print(f"✅ Registered function as tool: {tool.name}")
        return tool
    
    def register_skill(
        self,
        name: str,
        description: str,
        tools: list[str],
        instructions: str
    ):
        """Register a skill that combines tools for a specific task"""
        
        skill = create_skill(
            name=name,
            description=description,
            tools=tools,
            instructions=instructions
        )
        
        self.tools_manager.register_skill(skill)
        print(f"✅ Registered skill: {name}")
    
    def register_workflow(self, workflow: WorkflowDefinition):
        """Register a custom AI Developer Workflow"""
        self.orchestrator.register_workflow(workflow)
        print(f"✅ Registered workflow: {workflow.name}")
    
    # ============================================
    # EXECUTION METHODS
    # ============================================
    
    async def execute(
        self,
        task: str,
        role: str = "coder",
        context: str = "",
        use_closed_loop: bool = True
    ) -> TaskResult:
        """
        Execute a single task with an appropriate agent.
        
        Args:
            task: Description of what to do
            role: Agent role (coder, reviewer, tester, planner)
            context: Additional context to provide
            use_closed_loop: Whether to use self-correcting execution
        """
        from .agent_types import AgentRole
        
        role_enum = AgentRole(role.lower())
        
        # Get or create the appropriate agent
        agent = self.orchestrator._get_or_create_agent(role_enum)
        
        # Build full context
        full_context = f"{self._codebase_context}\n\n{context}" if self._codebase_context else context
        
        task_def = TaskDefinition(
            task_id=f"task_{len(self._execution_log)}",
            description=task,
            assigned_role=role_enum
        )
        
        # Execute
        if use_closed_loop and isinstance(agent, ClosedLoopAgent):
            result = await agent.execute_closed_loop(task_def)
        else:
            result = await agent.execute_task(task_def)
        
        # Log and update state
        self._execution_log.append({
            "task": task,
            "role": role,
            "result_status": result.status.value,
            "timestamp": datetime.now().isoformat()
        })
        
        self.state.completed_tasks += 1
        self._update_trust_metrics(result)
        
        return result
    
    async def orchestrate(
        self,
        request: str,
        context: str = ""
    ) -> dict:
        """
        Full multi-agent orchestration for complex requests.
        
        The orchestrator will:
        1. Analyze and plan the work
        2. Delegate to specialized agents
        3. Coordinate parallel execution
        4. Synthesize results
        """
        
        full_context = f"{self._codebase_context}\n\n{context}" if self._codebase_context else context
        
        result = await self.orchestrator.orchestrate(request, full_context)
        
        # Update state
        self.state.completed_tasks += result["metrics"]["total_tasks"]
        
        # Log
        self._execution_log.append({
            "type": "orchestration",
            "request": request,
            "tasks_completed": result["metrics"]["total_tasks"],
            "success": result["metrics"]["successful_tasks"] == result["metrics"]["total_tasks"],
            "timestamp": datetime.now().isoformat()
        })
        
        return result
    
    async def run_workflow(
        self,
        workflow_name: str,
        task: str,
        context: str = ""
    ) -> dict:
        """Run a predefined AI Developer Workflow"""
        
        full_context = f"{self._codebase_context}\n\n{context}" if self._codebase_context else context
        
        result = await self.orchestrator.run_workflow(workflow_name, task, full_context)
        
        # Log
        self._execution_log.append({
            "type": "workflow",
            "workflow": workflow_name,
            "task": task,
            "success": result["success"],
            "timestamp": datetime.now().isoformat()
        })
        
        return result
    
    # ============================================
    # TRUST & METRICS
    # ============================================
    
    def _update_trust_metrics(self, result: TaskResult):
        """Update trust metrics based on task result"""
        metrics = self.state.trust_metrics
        
        metrics.total_tasks += 1
        
        if result.status.value in ["success"]:
            metrics.successful_tasks += 1
        else:
            metrics.failed_tasks += 1
        
        metrics.total_tool_calls += result.tool_calls_made
        
        if result.tool_calls_made > metrics.longest_successful_chain:
            metrics.longest_successful_chain = result.tool_calls_made
        
        if result.needs_human_review:
            metrics.human_interventions += 1
    
    def get_trust_score(self) -> float:
        """Get current trust score (0-100)"""
        return self.state.trust_metrics.trust_score
    
    def get_metrics(self) -> dict:
        """Get comprehensive metrics"""
        return {
            "state": {
                "class": self.state.agentic_class.value,
                "grade": self.state.grade.value,
                "completed_tasks": self.state.completed_tasks
            },
            "trust": {
                "score": self.state.trust_metrics.trust_score,
                "success_rate": self.state.trust_metrics.success_rate,
                "total_tasks": self.state.trust_metrics.total_tasks,
                "tool_calls": self.state.trust_metrics.total_tool_calls,
                "longest_chain": self.state.trust_metrics.longest_successful_chain,
                "human_interventions": self.state.trust_metrics.human_interventions
            },
            "tools": self.tools_manager.get_tool_stats(),
            "orchestration": self.orchestrator.orchestration_metrics
        }
    
    # ============================================
    # PERSISTENCE
    # ============================================
    
    def save_state(self, filepath: Optional[str] = None):
        """Save agentic layer state to file"""
        if filepath is None:
            filepath = self.codebase_path / ".agentic_layer/state.json"
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        state_data = {
            "class": self.state.agentic_class.value,
            "grade": self.state.grade.value,
            "trust_metrics": self.state.trust_metrics.model_dump(),
            "execution_log": self._execution_log[-100:],  # Keep last 100
            "saved_at": datetime.now().isoformat()
        }
        
        filepath.write_text(json.dumps(state_data, indent=2))
        print(f"💾 State saved to {filepath}")
    
    def load_state(self, filepath: Optional[str] = None):
        """Load agentic layer state from file"""
        if filepath is None:
            filepath = self.codebase_path / ".agentic_layer/state.json"
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            print("No saved state found")
            return
        
        state_data = json.loads(filepath.read_text())
        
        self.state.agentic_class = AgenticClass(state_data.get("class", 1))
        self.state.grade = AgenticGrade(state_data.get("grade", 1))
        self.state.trust_metrics = TrustMetrics(**state_data.get("trust_metrics", {}))
        self._execution_log = state_data.get("execution_log", [])
        
        print(f"📂 State loaded from {filepath}")
    
    # ============================================
    # CONVENIENCE METHODS
    # ============================================
    
    async def quick_fix(self, description: str) -> TaskResult:
        """Quick bug fix with coder agent"""
        return await self.execute(description, role="coder", use_closed_loop=True)
    
    async def review(self, what: str) -> TaskResult:
        """Quick code review"""
        return await self.execute(f"Review: {what}", role="reviewer", use_closed_loop=False)
    
    async def plan(self, what: str) -> TaskResult:
        """Quick planning task"""
        return await self.execute(f"Plan: {what}", role="planner", use_closed_loop=False)
    
    def __repr__(self) -> str:
        return (
            f"AgenticLayer("
            f"class={self.state.agentic_class.value}, "
            f"grade={self.state.grade.value}, "
            f"trust={self.get_trust_score():.1f}%, "
            f"tasks={self.state.completed_tasks})"
        )


# ============================================
# CONVENIENCE FACTORY
# ============================================

def create_agentic_layer(
    codebase_path: Optional[str] = None,
    **kwargs
) -> AgenticLayer:
    """Factory function to create an agentic layer"""
    return AgenticLayer(codebase_path=codebase_path, **kwargs)