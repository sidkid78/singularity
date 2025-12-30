"""
Agent Tools Manager
====================
Prevents the "Grade 3 trap" where engineers overwhelm agents with too many tools.

Key principles:
- Curated tool selection per task type
- Token-efficient tool schemas
- Automatic tool filtering based on context
- Tool usage tracking for optimization
"""

from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json

from .agent_types import (
    ToolDefinition, 
    ToolParameter, 
    Skill, 
    AgentRole,
    TrustMetrics
)


@dataclass
class ToolUsageStats:
    """Track tool usage for optimization"""
    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_execution_time_ms: float = 0.0
    total_tokens_used: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.success_count / self.call_count


class AgentToolsManager:
    """
    Manages tools available to agents.
    
    Solves common problems:
    1. Too many tools overwhelming the agent
    2. Poor tool descriptions causing misuse
    3. Token waste from unused tools in context
    4. No tracking of tool effectiveness
    """
    
    def __init__(self, max_tools_per_request: int = 10):
        self.max_tools_per_request = max_tools_per_request
        self._tools: dict[str, ToolDefinition] = {}
        self._skills: dict[str, Skill] = {}
        self._python_functions: dict[str, Callable] = {}
        self._role_tools: dict[AgentRole, list[str]] = defaultdict(list)
        self._category_tools: dict[str, list[str]] = defaultdict(list)
        self._usage_stats: dict[str, ToolUsageStats] = defaultdict(ToolUsageStats)
        
        # Register default tools
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register commonly needed tools"""
        
        # File system tools
        self.register_tool(ToolDefinition(
            name="read_file",
            description="Read the contents of a file at the given path",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Absolute or relative path to the file"
                )
            ],
            category="filesystem",
            token_cost=50
        ))
        
        self.register_tool(ToolDefinition(
            name="write_file",
            description="Write content to a file, creating it if it doesn't exist",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path where the file should be written"
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Content to write to the file"
                )
            ],
            category="filesystem",
            token_cost=60,
            requires_confirmation=True
        ))
        
        self.register_tool(ToolDefinition(
            name="list_directory",
            description="List files and directories at the given path",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Directory path to list"
                ),
                ToolParameter(
                    name="recursive",
                    type="boolean",
                    description="Whether to list recursively",
                    required=False,
                    default=False
                )
            ],
            category="filesystem",
            token_cost=55
        ))
        
        # Code execution tools
        self.register_tool(ToolDefinition(
            name="execute_command",
            description="Execute a shell command and return the output",
            parameters=[
                ToolParameter(
                    name="command",
                    type="string",
                    description="The shell command to execute"
                ),
                ToolParameter(
                    name="working_directory",
                    type="string",
                    description="Directory to run the command in",
                    required=False
                )
            ],
            category="execution",
            token_cost=70,
            requires_confirmation=True
        ))
        
        self.register_tool(ToolDefinition(
            name="run_tests",
            description="Run test suite and return results",
            parameters=[
                ToolParameter(
                    name="test_path",
                    type="string",
                    description="Path to test file or directory",
                    required=False
                ),
                ToolParameter(
                    name="pattern",
                    type="string",
                    description="Test name pattern to match",
                    required=False
                )
            ],
            category="testing",
            token_cost=65
        ))
        
        # Search/analysis tools
        self.register_tool(ToolDefinition(
            name="search_codebase",
            description="Search for patterns in the codebase",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query or regex pattern"
                ),
                ToolParameter(
                    name="file_pattern",
                    type="string",
                    description="Glob pattern for files to search",
                    required=False,
                    default="**/*"
                )
            ],
            category="analysis",
            token_cost=60
        ))
        
        # Set up role-based tool mappings
        self._setup_role_mappings()
    
    def _setup_role_mappings(self):
        """Map tools to agent roles"""
        self._role_tools[AgentRole.CODER] = [
            "read_file", "write_file", "list_directory", 
            "execute_command", "search_codebase"
        ]
        self._role_tools[AgentRole.REVIEWER] = [
            "read_file", "list_directory", "search_codebase", "run_tests"
        ]
        self._role_tools[AgentRole.TESTER] = [
            "read_file", "write_file", "run_tests", "execute_command"
        ]
        self._role_tools[AgentRole.PLANNER] = [
            "read_file", "list_directory", "search_codebase"
        ]
        self._role_tools[AgentRole.DEBUGGER] = [
            "read_file", "write_file", "execute_command", 
            "run_tests", "search_codebase"
        ]
    
    def register_tool(
        self, 
        tool: ToolDefinition, 
        python_function: Optional[Callable] = None
    ):
        """Register a tool definition and optionally its implementation"""
        self._tools[tool.name] = tool
        self._category_tools[tool.category].append(tool.name)
        
        if python_function:
            self._python_functions[tool.name] = python_function
    
    def register_skill(self, skill: Skill):
        """Register a skill that combines tools for a specific task"""
        self._skills[skill.name] = skill
    
    def register_python_function(self, func: Callable):
        """
        Register a Python function directly as a tool.
        The function's docstring and type hints are used for the schema.
        """
        import inspect
        
        name = func.__name__
        doc = func.__doc__ or f"Execute {name}"
        sig = inspect.signature(func)
        
        parameters = []
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            # Infer type from annotation
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == list:
                    param_type = "array"
                elif param.annotation == dict:
                    param_type = "object"
            
            parameters.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=f"Parameter: {param_name}",
                required=param.default == inspect.Parameter.empty
            ))
        
        tool = ToolDefinition(
            name=name,
            description=doc.strip().split('\n')[0],  # First line of docstring
            parameters=parameters,
            category="custom"
        )
        
        self.register_tool(tool, func)
        return tool
    
    def get_tools_for_role(
        self, 
        role: AgentRole,
        additional_tools: list[str] = [],
        exclude_tools: list[str] = []
    ) -> list[ToolDefinition]:
        """Get curated tools for a specific agent role"""
        tool_names = set(self._role_tools.get(role, []))
        tool_names.update(additional_tools)
        tool_names -= set(exclude_tools)
        
        # Respect max tools limit
        if len(tool_names) > self.max_tools_per_request:
            # Prioritize by usage success rate
            sorted_tools = sorted(
                tool_names,
                key=lambda t: self._usage_stats[t].success_rate,
                reverse=True
            )
            tool_names = set(sorted_tools[:self.max_tools_per_request])
        
        return [self._tools[name] for name in tool_names if name in self._tools]
    
    def get_tools_for_task(
        self,
        task_description: str,
        role: AgentRole,
        required_tools: list[str] = []
    ) -> list[ToolDefinition]:
        """
        Intelligently select tools based on task description.
        This prevents tool overload by being selective.
        """
        selected = set(required_tools)
        
        # Add role-based tools
        selected.update(self._role_tools.get(role, []))
        
        # Task-based filtering using keywords
        task_lower = task_description.lower()
        
        # Only add file tools if task mentions files
        if not any(kw in task_lower for kw in ['file', 'read', 'write', 'create', 'modify']):
            selected -= {'write_file'}
        
        # Only add test tools if task mentions testing
        if not any(kw in task_lower for kw in ['test', 'verify', 'validate', 'check']):
            selected -= {'run_tests'}
        
        # Only add command execution if really needed
        if not any(kw in task_lower for kw in ['run', 'execute', 'command', 'build', 'install']):
            selected -= {'execute_command'}
        
        return [self._tools[name] for name in selected if name in self._tools]
    
    def get_skill(self, skill_name: str) -> Optional[Skill]:
        """Get a skill by name"""
        return self._skills.get(skill_name)
    
    def get_skill_tools(self, skill_name: str) -> list[ToolDefinition]:
        """Get all tools required by a skill"""
        skill = self._skills.get(skill_name)
        if not skill:
            return []
        return [self._tools[name] for name in skill.tools_required if name in self._tools]
    
    def to_gemini_tools(
        self, 
        tools: list[ToolDefinition]
    ) -> list[Callable] | list[dict]:
        """
        Convert tools to format for Gemini API.
        Returns Python functions if available, otherwise function declarations.
        """
        result = []
        
        for tool in tools:
            if tool.name in self._python_functions:
                # Automatic function calling with Python function
                result.append(self._python_functions[tool.name])
            else:
                # Manual function declaration
                result.append(tool.to_function_declaration())
        
        return result
    
    def record_tool_usage(
        self,
        tool_name: str,
        success: bool,
        execution_time_ms: float = 0.0,
        tokens_used: int = 0
    ):
        """Record tool usage for optimization"""
        stats = self._usage_stats[tool_name]
        stats.call_count += 1
        if success:
            stats.success_count += 1
        else:
            stats.failure_count += 1
        
        # Update rolling average execution time
        if stats.avg_execution_time_ms == 0:
            stats.avg_execution_time_ms = execution_time_ms
        else:
            stats.avg_execution_time_ms = (
                stats.avg_execution_time_ms * 0.9 + execution_time_ms * 0.1
            )
        
        stats.total_tokens_used += tokens_used
    
    def get_tool_stats(self) -> dict[str, dict]:
        """Get usage statistics for all tools"""
        return {
            name: {
                "call_count": stats.call_count,
                "success_rate": stats.success_rate,
                "avg_execution_time_ms": stats.avg_execution_time_ms,
                "total_tokens_used": stats.total_tokens_used
            }
            for name, stats in self._usage_stats.items()
            if stats.call_count > 0
        }
    
    def estimate_token_cost(self, tools: list[ToolDefinition]) -> int:
        """Estimate total token cost for a set of tools"""
        return sum(tool.token_cost for tool in tools)
    
    def get_high_risk_tools(self, tools: list[ToolDefinition]) -> list[str]:
        """Get tools that require confirmation before execution"""
        return [tool.name for tool in tools if tool.requires_confirmation]


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def create_tool(
    name: str,
    description: str,
    parameters: dict[str, tuple[str, str, bool]] = {},  # name: (type, desc, required)
    category: str = "general",
    requires_confirmation: bool = False
) -> ToolDefinition:
    """Convenience function to create a tool definition"""
    params = [
        ToolParameter(
            name=param_name,
            type=param_info[0],
            description=param_info[1],
            required=param_info[2] if len(param_info) > 2 else True
        )
        for param_name, param_info in parameters.items()
    ]
    
    return ToolDefinition(
        name=name,
        description=description,
        parameters=params,
        category=category,
        requires_confirmation=requires_confirmation
    )


def create_skill(
    name: str,
    description: str,
    tools: list[str],
    instructions: str,
    example: Optional[str] = None
) -> Skill:
    """Convenience function to create a skill"""
    return Skill(
        name=name,
        description=description,
        tools_required=tools,
        system_prompt=instructions,
        example_usage=example
    )