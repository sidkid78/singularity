"""
Orchestrator Agent
===================
Grade 5 "Lead Agent" that conducts a team of specialized agents.

Implements:
- Multi-agent orchestration
- AI Developer Workflows (ADWs)
- Task decomposition and delegation
- Parallel execution coordination
- Result synthesis
"""

from google import genai
from google.genai import types
from typing import Optional, Any
from pydantic import BaseModel
import asyncio
import uuid
import time

from .agent_types import (
    AgentRole,
    OrchestratorConfig,
    TaskDefinition,
    TaskResult,
    TaskStatus,
    ExecutionPlan,
    TrustMetrics,
    AgenticClass,
    AgenticGrade
)
from .tools_manager import AgentToolsManager
from .agents import (
    BaseAgent,
    ClosedLoopAgent,
    create_coder_agent,
    create_reviewer_agent,
    create_tester_agent,
    create_planner_agent
)


# ============================================
# STRUCTURED OUTPUTS FOR ORCHESTRATION
# ============================================

class TaskAssignment(BaseModel):
    """Single task assignment from orchestrator"""
    task_id: str
    description: str
    assigned_role: AgentRole
    priority: int = 1
    dependencies: list[str] = []
    tools_needed: list[str] = []
    estimated_complexity: str = "medium"  # low, medium, high


class OrchestrationPlan(BaseModel):
    """Full orchestration plan from lead agent"""
    plan_id: str
    goal_summary: str
    tasks: list[TaskAssignment]
    execution_order: list[str]  # Task IDs in order
    parallel_groups: list[list[str]] = []  # Groups that can run in parallel
    estimated_total_minutes: int = 5
    risks: list[str] = []


class WorkflowDefinition(BaseModel):
    """Definition of an AI Developer Workflow"""
    name: str
    description: str
    steps: list[str]  # Ordered step names
    agent_roles: list[AgentRole]  # Agents involved


# ============================================
# AI DEVELOPER WORKFLOWS (ADWs)
# ============================================

BUILT_IN_WORKFLOWS = {
    "plan_build": WorkflowDefinition(
        name="plan_build",
        description="Plan the work then build it",
        steps=["plan", "build"],
        agent_roles=[AgentRole.PLANNER, AgentRole.CODER]
    ),
    "plan_build_review": WorkflowDefinition(
        name="plan_build_review",
        description="Plan, build, and review the work",
        steps=["plan", "build", "review"],
        agent_roles=[AgentRole.PLANNER, AgentRole.CODER, AgentRole.REVIEWER]
    ),
    "plan_build_review_fix": WorkflowDefinition(
        name="plan_build_review_fix",
        description="Full cycle: plan, build, review, and fix issues",
        steps=["plan", "build", "review", "fix"],
        agent_roles=[AgentRole.PLANNER, AgentRole.CODER, AgentRole.REVIEWER, AgentRole.CODER]
    ),
    "test_driven": WorkflowDefinition(
        name="test_driven",
        description="Write tests first, then implement",
        steps=["plan", "write_tests", "build", "run_tests", "fix"],
        agent_roles=[AgentRole.PLANNER, AgentRole.TESTER, AgentRole.CODER, AgentRole.TESTER, AgentRole.CODER]
    )
}


# ============================================
# ORCHESTRATOR AGENT
# ============================================

class OrchestratorAgent(BaseAgent):
    """
    Lead Agent that orchestrates a team of specialized agents.
    
    This is the Grade 5 "Agentic Coding 2.0" implementation where:
    - You talk to the lead agent
    - Lead agent delegates to specialized agents
    - Specialized agents execute and return results
    - Lead agent synthesizes and delivers
    """
    
    ORCHESTRATOR_SYSTEM_PROMPT = """
You are a Lead Engineering Agent - an orchestrator that conducts a team of specialized agents.

<role>
You are a master planner and coordinator. You DO NOT write code yourself.
You analyze tasks, decompose them into specialized sub-tasks, and assign them to the right agents.
You monitor progress, handle failures, and synthesize results.
</role>

<available_agents>
- PLANNER: Architecture decisions, design patterns, task breakdown
- CODER: Implementation, code writing, bug fixes
- REVIEWER: Code review, quality checks, security audit
- TESTER: Test creation, validation, coverage analysis
- DOCUMENTER: Documentation, comments, README updates
- DEBUGGER: Bug investigation, root cause analysis
</available_agents>

<instructions>
1. **Analyze**: Understand the full scope of the request
2. **Decompose**: Break into atomic, specialized tasks
3. **Assign**: Match tasks to the right agent type
4. **Sequence**: Identify dependencies and parallelization opportunities
5. **Monitor**: Track progress and handle failures
6. **Synthesize**: Combine results into cohesive deliverable
</instructions>

<constraints>
- Never implement yourself - always delegate
- Maximize parallel execution where dependencies allow
- Flag high-risk tasks for human review
- If uncertain, ask for clarification before delegating
- Keep task descriptions specific and actionable
</constraints>

<output_format>
When creating a plan, use the structured format requested.
When synthesizing results, provide a clear summary of:
- What was accomplished
- Any issues encountered
- Recommendations for next steps
</output_format>
"""
    
    def __init__(
        self,
        tools_manager: AgentToolsManager,
        client: Optional[genai.Client] = None,
        max_concurrent_agents: int = 5
    ):
        config = OrchestratorConfig(
            role=AgentRole.ORCHESTRATOR,
            model="gemini-2.5-pro",
            system_instruction=self.ORCHESTRATOR_SYSTEM_PROMPT,
            thinking_budget=1024,
            max_concurrent_agents=max_concurrent_agents
        )
        
        super().__init__(config, tools_manager, client)
        
        self.max_concurrent_agents = max_concurrent_agents
        self.workflows = BUILT_IN_WORKFLOWS.copy()
        
        # Agent pool - lazy initialized
        self._agent_pool: dict[AgentRole, ClosedLoopAgent | BaseAgent] = {}
        
        # Track orchestration metrics
        self.orchestration_metrics = {
            "total_orchestrations": 0,
            "successful_orchestrations": 0,
            "total_tasks_delegated": 0,
            "avg_tasks_per_orchestration": 0.0
        }
    
    def _get_or_create_agent(self, role: AgentRole) -> ClosedLoopAgent | BaseAgent:
        """Get or create an agent for a specific role"""
        if role not in self._agent_pool:
            if role == AgentRole.CODER:
                self._agent_pool[role] = create_coder_agent(self.tools_manager, self.client)
            elif role == AgentRole.REVIEWER:
                self._agent_pool[role] = create_reviewer_agent(self.tools_manager, self.client)
            elif role == AgentRole.TESTER:
                self._agent_pool[role] = create_tester_agent(self.tools_manager, self.client)
            elif role == AgentRole.PLANNER:
                self._agent_pool[role] = create_planner_agent(self.tools_manager, self.client)
            else:
                # Default to coder for other roles
                self._agent_pool[role] = create_coder_agent(self.tools_manager, self.client)
        
        return self._agent_pool[role]
    
    async def create_plan(self, user_request: str, context: str = "") -> OrchestrationPlan:
        """Create an orchestration plan from user request"""
        
        planning_prompt = f"""
Create a detailed execution plan for the following request.
Break it down into specific tasks that can be assigned to specialized agents.

<request>
{user_request}
</request>

Generate a plan with:
- A clear goal summary
- Specific, actionable tasks with assigned roles
- Proper task ordering and parallel execution groups
- Time estimates and risk assessment
"""
        
        response = await self.execute(
            prompt=planning_prompt,
            context=context,
            response_schema=OrchestrationPlan
        )
        
        plan = OrchestrationPlan.model_validate_json(response)
        
        # Ensure plan_id is set
        if not plan.plan_id:
            plan.plan_id = str(uuid.uuid4())[:8]
        
        return plan
    
    async def execute_task(self, task: TaskAssignment) -> TaskResult:
        """Execute a single task using the appropriate agent"""
        
        agent = self._get_or_create_agent(task.assigned_role)
        
        task_def = TaskDefinition(
            task_id=task.task_id,
            description=task.description,
            assigned_role=task.assigned_role,
            priority=task.priority,
            dependencies=task.dependencies,
            tools_allowed=task.tools_needed
        )
        
        # Use closed-loop execution if agent supports it
        if isinstance(agent, ClosedLoopAgent):
            return await agent.execute_closed_loop(task_def)
        else:
            return await agent.execute_task(task_def)
    
    async def execute_parallel_group(
        self, 
        tasks: list[TaskAssignment]
    ) -> list[TaskResult]:
        """Execute a group of tasks in parallel"""
        
        # Limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent_agents)
        
        async def bounded_execute(task: TaskAssignment) -> TaskResult:
            async with semaphore:
                return await self.execute_task(task)
        
        results = await asyncio.gather(
            *[bounded_execute(task) for task in tasks],
            return_exceptions=True
        )
        
        # Convert exceptions to failed TaskResults
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(TaskResult(
                    task_id=tasks[i].task_id,
                    status=TaskStatus.FAILED,
                    agent_role=tasks[i].assigned_role,
                    output="",
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def orchestrate(
        self, 
        user_request: str,
        context: str = ""
    ) -> dict:
        """
        Full orchestration: Plan → Execute → Synthesize
        
        This is the main entry point for multi-agent orchestration.
        """
        
        start_time = time.time()
        self.orchestration_metrics["total_orchestrations"] += 1
        
        # Phase 1: Create plan
        print("🎯 Creating orchestration plan...")
        plan = await self.create_plan(user_request, context)
        print(f"📋 Plan: {plan.goal_summary}")
        print(f"📊 Tasks: {len(plan.tasks)}")
        
        # Phase 2: Execute tasks
        all_results: list[TaskResult] = []
        completed_task_ids: set[str] = set()
        task_map = {t.task_id: t for t in plan.tasks}
        
        # Execute parallel groups first
        if plan.parallel_groups:
            for group in plan.parallel_groups:
                group_tasks = [task_map[tid] for tid in group if tid in task_map]
                if group_tasks:
                    print(f"⚡ Executing parallel group: {[t.task_id for t in group_tasks]}")
                    group_results = await self.execute_parallel_group(group_tasks)
                    all_results.extend(group_results)
                    completed_task_ids.update(t.task_id for t in group_tasks)
        
        # Execute remaining tasks in order
        for task_id in plan.execution_order:
            if task_id in completed_task_ids:
                continue
            
            task = task_map.get(task_id)
            if not task:
                continue
            
            # Check dependencies
            deps_met = all(dep in completed_task_ids for dep in task.dependencies)
            if not deps_met:
                print(f"⏳ Waiting for dependencies: {task.dependencies}")
                continue
            
            print(f"🔧 Executing task: {task.task_id} ({task.assigned_role.value})")
            result = await self.execute_task(task)
            all_results.append(result)
            completed_task_ids.add(task_id)
        
        # Phase 3: Synthesize results
        synthesis = await self._synthesize_results(plan, all_results)
        
        execution_time = time.time() - start_time
        
        # Update metrics
        success_count = sum(1 for r in all_results if r.status == TaskStatus.SUCCESS)
        self.orchestration_metrics["total_tasks_delegated"] += len(all_results)
        
        if success_count == len(all_results):
            self.orchestration_metrics["successful_orchestrations"] += 1
        
        # Calculate running average
        total = self.orchestration_metrics["total_orchestrations"]
        prev_avg = self.orchestration_metrics["avg_tasks_per_orchestration"]
        self.orchestration_metrics["avg_tasks_per_orchestration"] = (
            prev_avg * (total - 1) / total + len(all_results) / total
        )
        
        return {
            "plan": plan.model_dump(),
            "results": [r.model_dump() for r in all_results],
            "synthesis": synthesis,
            "metrics": {
                "total_tasks": len(all_results),
                "successful_tasks": success_count,
                "failed_tasks": len(all_results) - success_count,
                "execution_time_seconds": round(execution_time, 2)
            }
        }
    
    async def _synthesize_results(
        self, 
        plan: OrchestrationPlan, 
        results: list[TaskResult]
    ) -> str:
        """Synthesize all results into a cohesive summary"""
        
        # Build context from results
        results_summary = "\n".join([
            f"Task {r.task_id} ({r.agent_role.value}): {r.status.value}\n"
            f"Output preview: {r.output[:200]}..." if len(r.output) > 200 else f"Output: {r.output}"
            for r in results
        ])
        
        synthesis_prompt = f"""
Synthesize the following task results into a cohesive summary.

<original_goal>
{plan.goal_summary}
</original_goal>

<task_results>
{results_summary}
</task_results>

Provide:
1. Executive summary of what was accomplished
2. Key deliverables produced
3. Any issues or failures that need attention
4. Recommended next steps
"""
        
        return await self.execute(prompt=synthesis_prompt)
    
    # ============================================
    # AI DEVELOPER WORKFLOWS
    # ============================================
    
    def register_workflow(self, workflow: WorkflowDefinition):
        """Register a custom workflow"""
        self.workflows[workflow.name] = workflow
    
    async def run_workflow(
        self,
        workflow_name: str,
        task_description: str,
        context: str = ""
    ) -> dict:
        """Run a predefined AI Developer Workflow"""
        
        workflow = self.workflows.get(workflow_name)
        if not workflow:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        print(f"🚀 Running workflow: {workflow.name}")
        print(f"📝 Steps: {workflow.steps}")
        
        results = []
        previous_output = ""
        
        for i, (step, role) in enumerate(zip(workflow.steps, workflow.agent_roles)):
            step_context = f"{context}\n\nPrevious step output:\n{previous_output}" if previous_output else context
            
            task = TaskAssignment(
                task_id=f"{workflow_name}_{step}_{i}",
                description=f"{step.upper()}: {task_description}",
                assigned_role=role,
                priority=i + 1
            )
            
            print(f"  Step {i+1}/{len(workflow.steps)}: {step} ({role.value})")
            result = await self.execute_task(task)
            results.append(result)
            
            if result.status == TaskStatus.SUCCESS:
                previous_output = result.output
            else:
                print(f"  ⚠️ Step failed: {result.error}")
                break
        
        return {
            "workflow": workflow_name,
            "steps_completed": len(results),
            "total_steps": len(workflow.steps),
            "results": [r.model_dump() for r in results],
            "success": all(r.status == TaskStatus.SUCCESS for r in results)
        }


# ============================================
# CONVENIENCE FACTORY
# ============================================

def create_orchestrator(
    tools_manager: Optional[AgentToolsManager] = None,
    client: Optional[genai.Client] = None,
    max_concurrent: int = 5
) -> OrchestratorAgent:
    """Factory function to create an orchestrator with defaults"""
    
    if tools_manager is None:
        tools_manager = AgentToolsManager()
    
    return OrchestratorAgent(
        tools_manager=tools_manager,
        client=client,
        max_concurrent_agents=max_concurrent
    )