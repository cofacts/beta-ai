"""
Agent tracing wrapper to enhance Google ADK agents with Langfuse conversation tracing.

This module provides wrappers and utilities to automatically trace agent interactions
while preserving the original ADK agent behavior.
"""

import functools
from typing import Any, Dict, Optional, Union
from langfuse_tracing import (
    trace_agent_interaction,
    trace_conversation_turn,
    record_agent_output,
    ConversationTracer,
    get_conversation_context
)


def trace_agent_call(agent_name: str, operation: str = "process"):
    """
    Decorator to trace agent calls with conversation grouping.
    
    Args:
        agent_name: Name of the agent being traced
        operation: Operation being performed
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract user input if available in arguments
            user_input = None
            if args and isinstance(args[0], str):
                user_input = args[0]
            elif 'message' in kwargs:
                user_input = kwargs['message']
            elif 'input' in kwargs:
                user_input = kwargs['input']
            
            with ConversationTracer(
                agent_name, 
                operation,
                input_data=user_input
            ) as tracer:
                try:
                    result = await func(*args, **kwargs)
                    tracer.record_output(result, success=True)
                    return result
                except Exception as e:
                    tracer.record_output(str(e), success=False)
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Extract user input if available in arguments
            user_input = None
            if args and isinstance(args[0], str):
                user_input = args[0]
            elif 'message' in kwargs:
                user_input = kwargs['message']
            elif 'input' in kwargs:
                user_input = kwargs['input']
            
            with ConversationTracer(
                agent_name, 
                operation,
                input_data=user_input
            ) as tracer:
                try:
                    result = func(*args, **kwargs)
                    tracer.record_output(result, success=True)
                    return result
                except Exception as e:
                    tracer.record_output(str(e), success=False)
                    raise
        
        # Return appropriate wrapper based on whether function is async
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def create_traced_llm_agent(original_agent, agent_name: str):
    """
    Create a traced version of an LlmAgent that maintains conversation context.
    
    Args:
        original_agent: The original Google ADK LlmAgent
        agent_name: Name for tracing purposes
        
    Returns:
        Enhanced agent with conversation tracing
    """
    
    # Store original methods that we want to trace
    original_process = getattr(original_agent, 'process', None)
    original_call = getattr(original_agent, '__call__', None)
    
    # Wrap the process method if it exists
    if original_process:
        @trace_agent_call(agent_name, "process")
        async def traced_process(*args, **kwargs):
            return await original_process(*args, **kwargs)
        
        # Replace the original method
        original_agent.process = traced_process
    
    # Wrap the __call__ method if it exists
    if original_call:
        @trace_agent_call(agent_name, "call")
        async def traced_call(*args, **kwargs):
            return await original_call(*args, **kwargs)
        
        # Replace the original method
        original_agent.__call__ = traced_call
    
    # Add conversation context methods
    def get_current_conversation_id():
        """Get the current conversation ID"""
        context = get_conversation_context()
        return context.get('conversation_id')
    
    def is_in_conversation():
        """Check if agent is currently in a conversation"""
        context = get_conversation_context()
        return context.get('has_active_conversation', False)
    
    # Attach helper methods to the agent
    original_agent.get_current_conversation_id = get_current_conversation_id
    original_agent.is_in_conversation = is_in_conversation
    original_agent._traced_agent_name = agent_name
    
    return original_agent


def trace_agent_delegation(
    delegating_agent: str,
    target_agent: str,
    task_description: str,
    input_data: Optional[Union[str, Dict[str, Any]]] = None
):
    """
    Create a span for agent-to-agent delegation within a conversation.
    
    Args:
        delegating_agent: Name of the agent doing the delegation
        target_agent: Name of the agent being called
        task_description: Description of the task being delegated
        input_data: Input data being passed to the target agent
        
    Returns:
        Span context manager for the delegation
    """
    return trace_agent_interaction(
        f"{delegating_agent}_to_{target_agent}",
        "delegate",
        input_data=input_data,
        delegating_agent=delegating_agent,
        target_agent=target_agent,
        task_description=task_description,
        operation_type="agent_delegation"
    )


def trace_multi_agent_workflow(
    workflow_name: str,
    agents_involved: list,
    user_input: Optional[str] = None
):
    """
    Create a span for multi-agent workflows (like the Cofacts fact-checking process).
    
    Args:
        workflow_name: Name of the workflow (e.g., "cofacts_factcheck")
        agents_involved: List of agent names that will participate
        user_input: Initial user input that triggered the workflow
        
    Returns:
        Span context manager for the entire workflow
    """
    return trace_agent_interaction(
        "multi_agent_workflow",
        workflow_name,
        input_data=user_input,
        workflow_name=workflow_name,
        agents_involved=','.join(agents_involved),
        operation_type="multi_agent_workflow"
    )


class TracedAgentOrchestrator:
    """
    Context manager for orchestrating multiple agents with conversation tracing.
    
    Usage:
        with TracedAgentOrchestrator("cofacts_factcheck", user_input) as orchestrator:
            # Delegate to investigator
            research_result = await orchestrator.delegate(
                ai_investigator, "research_claims", user_input
            )
            
            # Delegate to verifier
            verification_result = await orchestrator.delegate(
                ai_verifier, "verify_sources", research_result
            )
            
            # Record final output
            orchestrator.record_workflow_output(final_result)
    """
    
    def __init__(self, workflow_name: str, user_input: Optional[str] = None):
        self.workflow_name = workflow_name
        self.user_input = user_input
        self.span = None
        self.delegations = []
    
    def __enter__(self):
        self.span = trace_multi_agent_workflow(
            self.workflow_name,
            [],  # Will be populated as agents are used
            self.user_input
        )
        self.span.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Record error in span
            self.span.set_attribute("workflow.error.type", exc_type.__name__)
            self.span.set_attribute("workflow.error.message", str(exc_val))
        
        # Record summary of delegations
        self.span.set_attribute("workflow.delegations_count", len(self.delegations))
        if self.delegations:
            self.span.set_attribute("workflow.agents_used", ','.join(self.delegations))
        
        self.span.__exit__(exc_type, exc_val, exc_tb)
    
    async def delegate(self, agent, task_name: str, input_data: Any = None):
        """
        Delegate a task to an agent with tracing.
        
        Args:
            agent: The agent to delegate to
            task_name: Name of the task being delegated
            input_data: Input data for the agent
            
        Returns:
            Result from the agent
        """
        agent_name = getattr(agent, '_traced_agent_name', agent.name if hasattr(agent, 'name') else 'unknown_agent')
        
        # Track this delegation
        if agent_name not in self.delegations:
            self.delegations.append(agent_name)
        
        with trace_agent_delegation(
            self.workflow_name,
            agent_name,
            task_name,
            input_data
        ) as delegation_span:
            try:
                # Call the agent (this should already be traced by the agent's own wrapper)
                result = await agent(input_data) if hasattr(agent, '__call__') else await agent.process(input_data)
                
                # Record successful delegation
                delegation_span.set_attribute("delegation.success", True)
                record_agent_output(delegation_span, result, success=True)
                
                return result
            except Exception as e:
                # Record failed delegation
                delegation_span.set_attribute("delegation.success", False)
                delegation_span.set_attribute("delegation.error", str(e))
                record_agent_output(delegation_span, str(e), success=False)
                raise
    
    def record_workflow_output(self, output: Any):
        """Record the final output of the workflow"""
        if self.span:
            record_agent_output(self.span, output, success=True)
    
    def record_intermediate_result(self, step_name: str, result: Any):
        """Record intermediate results in the workflow"""
        if self.span:
            self.span.set_attribute(f"workflow.step.{step_name}", str(result)[:200])


def enhance_agents_with_tracing(agents_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance a dictionary of agents with conversation tracing.
    
    Args:
        agents_dict: Dictionary mapping agent names to agent instances
        
    Returns:
        Dictionary with traced agents
    """
    traced_agents = {}
    
    for agent_name, agent in agents_dict.items():
        traced_agents[agent_name] = create_traced_llm_agent(agent, agent_name)
    
    return traced_agents