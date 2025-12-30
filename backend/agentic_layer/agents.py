"""
Agent Classes
==============
Base agent and specialized agent implementations using google-genai SDK.

Implements:
- Grade 2: Sub-agents with specialized roles
- Grade 4: Closed-loop execution (Request → Validate → Resolve)
"""

from google import genai
from google.genai import types
from typing import Optional, Any
from pydantic import BaseModel
import asyncio
import time
import uuid

from .agent_types import (
    AgentRole,
    AgentConfig,
    TaskDefinition,
    TaskResult,
    TaskStatus,
    FeedbackLoop,
    FeedbackLoopPhase,
    ValidationResult,
    TrustMetrics
)
from .tools_manager import AgentToolsManager


# ============================================
# BASE AGENT
# ============================================

class BaseAgent:
    """
    Base agent class using google-genai SDK.
    
    Supports:
    - Automatic function calling
    - Structured outputs
    - Thinking budgets (for Gemini 2.5)
    - Tool execution tracking
    """
    
    def __init__(
        self,
        config: AgentConfig,
        tools_manager: AgentToolsManager,
        client: Optional[genai.Client] = None
    ):
        self.config = config
        self.tools_manager = tools_manager
        self.client = client or genai.Client()
        self.metrics = TrustMetrics()
        self._execution_history: list[dict] = []
    
    def _get_generate_config(
        self,
        tools: list = [],
        response_schema: Optional[type[BaseModel]] = None,
        override_thinking_budget: Optional[int] = None
    ) -> types.GenerateContentConfig:
        """Build GenerateContentConfig with proper settings"""
        
        config_dict = {
            "system_instruction": self.config.system_instruction,
            "temperature": self.config.temperature,
        }
        
        # Add tools if provided
        if tools:
            config_dict["tools"] = tools
        
        # Add structured output if schema provided
        if response_schema:
            config_dict["response_mime_type"] = "application/json"
            config_dict["response_schema"] = response_schema
        
        # Add thinking budget for Gemini 2.5 models
        thinking_budget = override_thinking_budget or self.config.thinking_budget
        if thinking_budget > 0 and "2.5" in self.config.model:
            config_dict["thinking_config"] = types.ThinkingConfig(
                thinking_budget=thinking_budget
            )
        
        # Add max tokens if specified
        if self.config.max_output_tokens:
            config_dict["max_output_tokens"] = self.config.max_output_tokens
        
        return types.GenerateContentConfig(**config_dict)
    
    async def execute(
        self,
        prompt: str,
        context: str = "",
        tools: list = [],
        response_schema: Optional[type[BaseModel]] = None
    ) -> str:
        """Execute a single generation with optional tools"""
        
        # Build contents
        if context:
            contents = f"<context>\n{context}\n</context>\n\n<task>\n{prompt}\n</task>"
        else:
            contents = prompt
        
        config = self._get_generate_config(
            tools=tools,
            response_schema=response_schema
        )
        
        start_time = time.time()
        
        response = await self.client.aio.models.generate_content(
            model=self.config.model,
            contents=contents,
            config=config
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        # Track execution
        self._execution_history.append({
            "prompt_preview": prompt[:100],
            "model": self.config.model,
            "execution_time_ms": execution_time,
            "had_tools": len(tools) > 0,
            "timestamp": time.time()
        })
        
        return response.text
    
    async def execute_with_tools(
        self,
        prompt: str,
        context: str = "",
        additional_tools: list[str] = [],
        max_tool_calls: int = 10
    ) -> tuple[str, list[dict]]:
        """
        Execute with automatic function calling enabled.
        Returns (response_text, tool_calls_made)
        """
        
        # Get tools for this agent's role
        tool_defs = self.tools_manager.get_tools_for_role(
            self.config.role,
            additional_tools=additional_tools
        )
        
        # Convert to Gemini format
        gemini_tools = self.tools_manager.to_gemini_tools(tool_defs)
        
        config = self._get_generate_config(tools=gemini_tools)
        
        # Enable automatic function calling with limit
        config.automatic_function_calling = types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=max_tool_calls
        )
        
        if context:
            contents = f"<context>\n{context}\n</context>\n\n<task>\n{prompt}\n</task>"
        else:
            contents = prompt
        
        start_time = time.time()
        tool_calls = []
        
        response = await self.client.aio.models.generate_content(
            model=self.config.model,
            contents=contents,
            config=config
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        # Track tool usage
        if hasattr(response, 'function_calls') and response.function_calls:
            for fc in response.function_calls:
                tool_calls.append({
                    "name": fc.name,
                    "args": dict(fc.args) if fc.args else {}
                })
                # Record usage (assume success for automatic calling)
                self.tools_manager.record_tool_usage(
                    fc.name,
                    success=True,
                    execution_time_ms=execution_time / len(response.function_calls)
                )
        
        # Update metrics
        self.metrics.total_tool_calls += len(tool_calls)
        if len(tool_calls) > self.metrics.longest_successful_chain:
            self.metrics.longest_successful_chain = len(tool_calls)
        
        return response.text, tool_calls
    
    async def execute_task(self, task: TaskDefinition) -> TaskResult:
        """Execute a task definition and return structured result"""
        
        start_time = time.time()
        self.metrics.total_tasks += 1
        
        try:
            # Get task-specific tools
            tool_defs = self.tools_manager.get_tools_for_task(
                task.description,
                self.config.role,
                required_tools=task.tools_allowed
            )
            gemini_tools = self.tools_manager.to_gemini_tools(tool_defs)
            
            response_text, tool_calls = await self.execute_with_tools(
                prompt=task.description,
                additional_tools=task.tools_allowed,
                max_tool_calls=task.max_iterations
            )
            
            execution_time = (time.time() - start_time) * 1000
            self.metrics.successful_tasks += 1
            
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.SUCCESS,
                agent_role=self.config.role,
                output=response_text,
                tool_calls_made=len(tool_calls),
                execution_time_ms=execution_time,
                confidence_score=0.85  # Could be computed from response
            )
            
        except Exception as e:
            self.metrics.failed_tasks += 1
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                agent_role=self.config.role,
                output="",
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )


# ============================================
# CLOSED-LOOP AGENT (Grade 4)
# ============================================

class ClosedLoopAgent(BaseAgent):
    """
    Agent with closed-loop execution capability.
    Implements Request → Validate → Resolve pattern.
    """
    
    # Validation prompt template
    VALIDATION_PROMPT = """
    Review the following output and determine if it successfully completes the task.
    
    <original_task>
    {task}
    </original_task>
    
    <output_to_validate>
    {output}
    </output_to_validate>
    
    Evaluate:
    1. Does the output fully address the task requirements?
    2. Are there any errors, issues, or missing elements?
    3. What is your confidence level (0.0-1.0)?
    
    Be critical but fair. Only mark as valid if truly complete.
    """
    
    # Resolution prompt template  
    RESOLUTION_PROMPT = """
    The previous attempt did not fully complete the task. Fix the issues.
    
    <original_task>
    {task}
    </original_task>
    
    <previous_output>
    {previous_output}
    </previous_output>
    
    <issues_found>
    {issues}
    </issues_found>
    
    <suggestions>
    {suggestions}
    </suggestions>
    
    Provide a corrected, complete solution that addresses all issues.
    """
    
    async def validate_output(
        self, 
        task: str, 
        output: str
    ) -> ValidationResult:
        """Validate task output"""
        
        # Use structured output for validation
        validation_prompt = self.VALIDATION_PROMPT.format(
            task=task,
            output=output
        )
        
        response = await self.execute(
            prompt=validation_prompt,
            response_schema=ValidationResult
        )
        
        return ValidationResult.model_validate_json(response)
    
    async def execute_closed_loop(
        self,
        task: TaskDefinition,
        max_iterations: int = 5
    ) -> TaskResult:
        """
        Execute task with closed-loop validation.
        Keeps iterating until valid or max iterations reached.
        """
        
        loop = FeedbackLoop(
            loop_id=str(uuid.uuid4())[:8],
            task_id=task.task_id,
            current_phase=FeedbackLoopPhase.REQUEST,
            max_iterations=max_iterations
        )
        
        start_time = time.time()
        current_output = ""
        all_tool_calls = 0
        
        while loop.iteration <= loop.max_iterations:
            # Phase 1: REQUEST - Execute the task
            if loop.current_phase == FeedbackLoopPhase.REQUEST:
                response_text, tool_calls = await self.execute_with_tools(
                    prompt=task.description if loop.iteration == 1 else self.RESOLUTION_PROMPT.format(
                        task=task.description,
                        previous_output=current_output,
                        issues="\n".join(loop.history[-1].get("issues", [])) if loop.history else "",
                        suggestions="\n".join(loop.history[-1].get("suggestions", [])) if loop.history else ""
                    ),
                    additional_tools=task.tools_allowed
                )
                
                current_output = response_text
                all_tool_calls += len(tool_calls)
                
                loop.record_iteration(FeedbackLoopPhase.REQUEST, {
                    "output_preview": current_output[:200],
                    "tool_calls": len(tool_calls)
                })
                
                loop.current_phase = FeedbackLoopPhase.VALIDATE
            
            # Phase 2: VALIDATE - Check the output
            elif loop.current_phase == FeedbackLoopPhase.VALIDATE:
                validation = await self.validate_output(
                    task=task.description,
                    output=current_output
                )
                
                loop.record_iteration(FeedbackLoopPhase.VALIDATE, {
                    "is_valid": validation.is_valid,
                    "confidence": validation.confidence,
                    "issues": validation.issues,
                    "suggestions": validation.suggestions
                })
                
                if validation.is_valid and validation.confidence >= 0.8:
                    # Success! Exit the loop
                    loop.current_phase = FeedbackLoopPhase.RESOLVE
                    break
                elif validation.requires_retry and loop.iteration < loop.max_iterations:
                    # Need to retry
                    loop.iteration += 1
                    loop.current_phase = FeedbackLoopPhase.REQUEST
                else:
                    # Can't or won't retry
                    break
        
        execution_time = (time.time() - start_time) * 1000
        
        # Determine final status
        final_validation = loop.history[-1] if loop.history else {}
        is_success = final_validation.get("is_valid", False)
        
        # Update metrics
        self.metrics.total_tasks += 1
        if is_success:
            self.metrics.successful_tasks += 1
            self.metrics.successful_tool_chains += 1
        else:
            self.metrics.failed_tasks += 1
        
        # Track average iterations
        total_iterations = sum(1 for h in loop.history if h.get("phase") == "request")
        if self.metrics.avg_iterations_to_success == 0:
            self.metrics.avg_iterations_to_success = total_iterations
        else:
            self.metrics.avg_iterations_to_success = (
                self.metrics.avg_iterations_to_success * 0.9 + total_iterations * 0.1
            )
        
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.SUCCESS if is_success else TaskStatus.NEEDS_REVIEW,
            agent_role=self.config.role,
            output=current_output,
            tool_calls_made=all_tool_calls,
            iterations_used=loop.iteration,
            execution_time_ms=execution_time,
            confidence_score=final_validation.get("confidence", 0.0),
            needs_human_review=not is_success,
            feedback_notes=[
                f"Iteration {i+1}: {h.get('phase', 'unknown')}"
                for i, h in enumerate(loop.history)
            ]
        )


# ============================================
# SPECIALIZED AGENTS
# ============================================

def create_coder_agent(
    tools_manager: AgentToolsManager,
    client: Optional[genai.Client] = None
) -> ClosedLoopAgent:
    """Factory for creating a coder agent"""
    
    config = AgentConfig(
        role=AgentRole.CODER,
        model="gemini-3-flash-review",
        system_instruction="""
You are a specialized Coding Agent. You write clean, efficient, production-ready code.

<role>
Expert software engineer focused on implementation.
</role>

<instructions>
1. Read and understand the task completely before coding
2. Write clean, well-structured code with proper error handling
3. Include type hints and docstrings
4. Keep functions focused and small (single responsibility)
5. Follow the existing code style in the codebase
</instructions>

<constraints>
- Always handle errors gracefully
- Use meaningful variable and function names
- Add comments only when logic is complex
- Test your changes mentally before submitting
</constraints>

<output_format>
Provide code in markdown code blocks with language specified.
Explain key decisions briefly after the code.
</output_format>
""",
        tools=["read_file", "write_file", "list_directory", "execute_command", "search_codebase"],
        thinking_budget=0,  # Fast execution
        temperature=0.3  # More deterministic for code
    )
    
    return ClosedLoopAgent(config, tools_manager, client)


def create_reviewer_agent(
    tools_manager: AgentToolsManager,
    client: Optional[genai.Client] = None
) -> ClosedLoopAgent:
    """Factory for creating a reviewer agent"""
    
    config = AgentConfig(
        role=AgentRole.REVIEWER,
        model="gemini-flash-latest",
        system_instruction="""
You are a Code Review Agent. You identify bugs, security issues, and improvements.

<role>
Expert code reviewer focused on quality and correctness.
</role>

<instructions>
1. Review code thoroughly for bugs and logic errors
2. Check for security vulnerabilities
3. Evaluate code structure and maintainability
4. Suggest specific improvements with examples
</instructions>

<output_format>
Categorize findings:
- CRITICAL: Must fix before merge (security, bugs, data loss)
- WARNING: Should fix (performance, maintainability)
- SUGGESTION: Nice to have (style, minor improvements)
- APPROVED: No significant issues found

For each finding, provide:
- Location (file:line if possible)
- Description of the issue
- Suggested fix with code example
</output_format>
""",
        tools=["read_file", "list_directory", "search_codebase", "run_tests"],
        thinking_budget=256,  # Some reasoning for review
        temperature=0.5
    )
    
    return ClosedLoopAgent(config, tools_manager, client)


def create_tester_agent(
    tools_manager: AgentToolsManager,
    client: Optional[genai.Client] = None
) -> ClosedLoopAgent:
    """Factory for creating a tester agent"""
    
    config = AgentConfig(
        role=AgentRole.TESTER,
        model="gemini-flash-latest",
        system_instruction="""
You are a Testing Agent. You write comprehensive tests and validate functionality.

<role>
Expert test engineer focused on coverage and edge cases.
</role>

<instructions>
1. Analyze the code to identify all testable behaviors
2. Write tests for happy path scenarios
3. Write tests for edge cases and error conditions
4. Ensure tests are independent and repeatable
5. Run tests and fix any failures
</instructions>

<constraints>
- Use pytest conventions
- Each test should test one thing
- Use descriptive test names: test_<function>_<scenario>_<expected>
- Include setup and teardown when needed
</constraints>

<output_format>
Provide test code in pytest format.
List coverage summary:
- Functions tested
- Edge cases covered
- Any untestable areas
</output_format>
""",
        tools=["read_file", "write_file", "run_tests", "execute_command"],
        thinking_budget=0,
        temperature=0.3
    )
    
    return ClosedLoopAgent(config, tools_manager, client)


def create_planner_agent(
    tools_manager: AgentToolsManager,
    client: Optional[genai.Client] = None
) -> BaseAgent:
    """Factory for creating a planner agent (no closed-loop needed)"""
    
    config = AgentConfig(
        role=AgentRole.PLANNER,
        model="gemini-2.5-pro",  # Pro for better planning
        system_instruction="""
You are a Planning Agent. You analyze requirements and create detailed execution plans.

<role>
Expert software architect focused on planning and decomposition.
</role>

<instructions>
1. Analyze the full scope of the request
2. Identify all components, dependencies, and constraints
3. Break down into atomic, actionable tasks
4. Determine task order and parallelization opportunities
5. Estimate complexity and time for each task
</instructions>

<output_format>
Provide a structured plan:
1. Goal Summary: One sentence overview
2. Tasks: Numbered list with:
   - Description
   - Assigned role (CODER, REVIEWER, TESTER, etc.)
   - Dependencies (other task numbers)
   - Complexity (low/medium/high)
3. Execution Order: Which tasks first, what can run parallel
4. Risks: What could go wrong and mitigation
</output_format>
""",
        tools=["read_file", "list_directory", "search_codebase"],
        thinking_budget=1024,  # High thinking for planning
        temperature=0.7
    )
    
    return BaseAgent(config, tools_manager, client)