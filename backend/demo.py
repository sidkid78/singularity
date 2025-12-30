#!/usr/bin/env python3
"""
Agentic Layer Framework - Demo Script
======================================
Demonstrates all major features of the agentic layer framework.

Usage:
    export GEMINI_API_KEY="your-api-key"
    python demo.py

Features demonstrated:
1. Basic agent execution
2. Closed-loop self-correcting execution
3. Custom tool registration
4. Multi-agent orchestration
5. AI Developer Workflows
6. Trust measurement
7. Grade assessment
"""

import asyncio
import os
from pathlib import Path

# Ensure we can import the package
import sys
sys.path.insert(0, str(Path(__file__).parent))

from agentic_layer import (
    AgenticLayer,
    create_agentic_layer,
    WorkflowDefinition,
    AgentRole,
    create_tool,
)


async def demo_basic_execution(layer: AgenticLayer):
    """Demo 1: Basic task execution"""
    print("\n" + "="*60)
    print("DEMO 1: Basic Task Execution")
    print("="*60)
    
    # Simple execution with coder agent
    result = await layer.execute(
        task="Write a Python function that calculates fibonacci numbers recursively with memoization",
        role="coder"
    )
    
    print(f"\nStatus: {result.status.value}")
    print(f"Tool calls: {result.tool_calls_made}")
    print(f"Iterations: {result.iterations_used}")
    print(f"Output preview:\n{result.output[:500]}...")


async def demo_closed_loop(layer: AgenticLayer):
    """Demo 2: Closed-loop self-correcting execution"""
    print("\n" + "="*60)
    print("DEMO 2: Closed-Loop Execution (Request → Validate → Resolve)")
    print("="*60)
    
    result = await layer.execute(
        task="""Write a Python class for a simple REST API client with:
        1. GET, POST, PUT, DELETE methods
        2. Automatic retry with exponential backoff
        3. Request/response logging
        4. Proper error handling
        Include type hints and docstrings.""",
        role="coder",
        use_closed_loop=True  # Enable self-correction
    )
    
    print(f"\nStatus: {result.status.value}")
    print(f"Iterations used: {result.iterations_used}")
    print(f"Confidence: {result.confidence_score:.2f}")
    print(f"Needs review: {result.needs_human_review}")
    
    if result.feedback_notes:
        print(f"Feedback loop history:")
        for note in result.feedback_notes:
            print(f"  - {note}")


async def demo_custom_tools(layer: AgenticLayer):
    """Demo 3: Registering custom tools"""
    print("\n" + "="*60)
    print("DEMO 3: Custom Tool Registration")
    print("="*60)
    
    # Register a custom tool with manual definition
    layer.register_tool(
        name="get_project_config",
        description="Get configuration values from the project's config file",
        parameters={
            "key": ("string", "Configuration key to retrieve", True),
            "default": ("string", "Default value if key not found", False)
        },
        category="custom"
    )
    
    # Register a Python function directly as a tool
    def analyze_dependencies(package_json_path: str) -> dict:
        """Analyze project dependencies from package.json or requirements.txt"""
        # Mock implementation
        return {
            "total_dependencies": 42,
            "outdated": 5,
            "security_issues": 0
        }
    
    layer.register_function(analyze_dependencies)
    
    # Register a skill that combines tools
    layer.register_skill(
        name="setup_new_feature",
        description="Set up boilerplate for a new feature",
        tools=["read_file", "write_file", "get_project_config"],
        instructions="""
        When setting up a new feature:
        1. Read the project config to understand conventions
        2. Create the necessary directory structure
        3. Generate boilerplate files following project patterns
        4. Update any index/barrel files
        """
    )
    
    print("✅ Custom tools and skills registered")
    print(f"   Tools available: {list(layer.tools_manager._tools.keys())}")


async def demo_orchestration(layer: AgenticLayer):
    """Demo 4: Multi-agent orchestration"""
    print("\n" + "="*60)
    print("DEMO 4: Multi-Agent Orchestration")
    print("="*60)
    
    result = await layer.orchestrate(
        request="""Build a user authentication module with:
        1. User model with email, password hash, created_at
        2. Registration endpoint with validation
        3. Login endpoint with JWT token generation
        4. Password reset flow
        5. Unit tests for all endpoints"""
    )
    
    print(f"\n📋 Plan: {result['plan']['goal_summary']}")
    print(f"📊 Tasks executed: {result['metrics']['total_tasks']}")
    print(f"✅ Successful: {result['metrics']['successful_tasks']}")
    print(f"⏱️  Duration: {result['metrics']['execution_time_seconds']}s")
    
    print("\n📝 Task breakdown:")
    for task in result['plan']['tasks']:
        print(f"   - [{task['assigned_role']}] {task['description'][:60]}...")
    
    print(f"\n📄 Synthesis:\n{result['synthesis'][:500]}...")


async def demo_workflows(layer: AgenticLayer):
    """Demo 5: AI Developer Workflows"""
    print("\n" + "="*60)
    print("DEMO 5: AI Developer Workflows")
    print("="*60)
    
    # List available workflows
    print("\nAvailable workflows:")
    for name, workflow in layer.orchestrator.workflows.items():
        print(f"   - {name}: {workflow.description}")
        print(f"     Steps: {' → '.join(workflow.steps)}")
    
    # Register a custom workflow
    layer.register_workflow(WorkflowDefinition(
        name="security_audit",
        description="Security-focused development workflow",
        steps=["plan", "build", "security_review", "fix", "test"],
        agent_roles=[
            AgentRole.PLANNER,
            AgentRole.CODER,
            AgentRole.REVIEWER,
            AgentRole.CODER,
            AgentRole.TESTER
        ]
    ))
    
    # Run a workflow
    result = await layer.run_workflow(
        workflow_name="plan_build_review",
        task="Add rate limiting middleware to the API"
    )
    
    print(f"\n🚀 Workflow: {result['workflow']}")
    print(f"📊 Steps completed: {result['steps_completed']}/{result['total_steps']}")
    print(f"✅ Success: {result['success']}")


async def demo_trust_metrics(layer: AgenticLayer):
    """Demo 6: Trust measurement"""
    print("\n" + "="*60)
    print("DEMO 6: Trust Measurement & Metrics")
    print("="*60)
    
    metrics = layer.get_metrics()
    
    print(f"\n📊 Agentic Layer State:")
    print(f"   Class: {metrics['state']['class']}")
    print(f"   Grade: {metrics['state']['grade']}")
    print(f"   Tasks completed: {metrics['state']['completed_tasks']}")
    
    print(f"\n🎯 Trust Metrics:")
    print(f"   Trust Score: {metrics['trust']['score']:.1f}/100")
    print(f"   Success Rate: {metrics['trust']['success_rate']*100:.1f}%")
    print(f"   Total Tool Calls: {metrics['trust']['tool_calls']}")
    print(f"   Longest Successful Chain: {metrics['trust']['longest_chain']}")
    print(f"   Human Interventions: {metrics['trust']['human_interventions']}")
    
    print(f"\n🔧 Tool Usage Stats:")
    for tool_name, stats in metrics['tools'].items():
        print(f"   {tool_name}: {stats['call_count']} calls, {stats['success_rate']*100:.0f}% success")
    
    print(f"\n🎭 Orchestration Stats:")
    for key, value in metrics['orchestration'].items():
        print(f"   {key}: {value}")


async def demo_grade_assessment(layer: AgenticLayer):
    """Demo 7: Grade assessment"""
    print("\n" + "="*60)
    print("DEMO 7: Agentic Layer Grade Assessment")
    print("="*60)
    
    assessment = layer.assess_grade()
    
    print(f"\n📊 Current Grade: {assessment['current_grade']}")
    print(f"🎯 Suggested Grade: {assessment['suggested_grade']}")
    
    print(f"\n✅ Capabilities:")
    for cap, value in assessment['capabilities'].items():
        status = "✓" if value else "✗"
        print(f"   {status} {cap}: {value}")
    
    print(f"\n💡 Recommendations:")
    for rec in assessment['recommendations']:
        print(f"   → {rec}")


async def main():
    """Run all demos"""
    print("="*60)
    print("AGENTIC LAYER FRAMEWORK - DEMO")
    print("="*60)
    
    # Check for API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("\n⚠️  Warning: GEMINI_API_KEY not set")
        print("   Set it with: export GEMINI_API_KEY='your-key'")
        print("   Running in demo mode (some features may not work)\n")
    
    # Create the agentic layer
    layer = create_agentic_layer(
        codebase_path=".",  # Current directory
        max_concurrent_agents=5
    )
    
    print(f"\n🚀 Created: {layer}")
    
    try:
        # Run demos (comment out to run specific ones)
        # await demo_basic_execution(layer)
        # await demo_closed_loop(layer)
        await demo_custom_tools(layer)
        # await demo_orchestration(layer)
        # await demo_workflows(layer)
        await demo_trust_metrics(layer)
        await demo_grade_assessment(layer)
        
        # Save state
        # layer.save_state()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print(f"\nFinal state: {layer}")


if __name__ == "__main__":
    asyncio.run(main())