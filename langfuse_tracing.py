"""
Langfuse conversation tracing utilities for Google ADK agents.

This module provides conversation-aware tracing that groups all agent interactions,
tool calls, and outputs by conversation ID for proper Langfuse session management.
"""

import uuid
import json
from contextvars import ContextVar
from typing import Any, Dict, Optional, Union
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# Context variable to store conversation ID throughout request lifecycle
conversation_id_var: ContextVar[Optional[str]] = ContextVar('conversation_id', default=None)

# Get the tracer that's already configured in main.py
def get_tracer():
    """Get the existing ADK application tracer"""
    return trace.get_tracer("adk_application_tracer")


def generate_conversation_id() -> str:
    """Generate a unique conversation ID for grouping traces in Langfuse"""
    return str(uuid.uuid4())


def get_or_create_conversation_id() -> str:
    """Get existing conversation ID or create new one if none exists"""
    conv_id = conversation_id_var.get()
    if not conv_id:
        conv_id = generate_conversation_id()
        conversation_id_var.set(conv_id)
    return conv_id


def set_conversation_id(conversation_id: str) -> None:
    """Set the conversation ID for the current context"""
    conversation_id_var.set(conversation_id)


def detect_conversation_start(user_input: str) -> bool:
    """Detect if this input indicates the start of a new conversation"""
    if not user_input:
        return False
    
    user_input_lower = user_input.lower()
    
    # Cofacts conversations typically start with URLs
    if "cofacts.tw/article/" in user_input:
        return True
    
    # HackMD conversations start with HackMD URLs  
    if "hackmd.io" in user_input:
        return True
    
    # Other conversation starters
    conversation_starters = [
        "help me", "analyze", "fact-check", "investigate", 
        "review", "check this", "what do you think"
    ]
    
    return any(starter in user_input_lower for starter in conversation_starters)


def create_conversation_span(
    operation_name: str,
    input_data: Optional[Union[str, Dict[str, Any]]] = None,
    **additional_attributes
):
    """
    Create an OpenTelemetry span with conversation context for Langfuse grouping.
    
    Args:
        operation_name: Name of the operation/span
        input_data: User input or operation input data
        **additional_attributes: Additional span attributes
    """
    tracer = get_tracer()
    conversation_id = get_or_create_conversation_id()
    
    span = tracer.start_span(operation_name)
    
    # Core conversation attributes for Langfuse grouping
    span.set_attribute("conversation.id", conversation_id)
    span.set_attribute("conversation.session_id", conversation_id)  # Langfuse session grouping
    span.set_attribute("service.name", "adk_langfuse_service")
    
    # Add input data if provided
    if input_data:
        if isinstance(input_data, str):
            span.set_attribute("input.content", input_data[:1000])  # Truncate long inputs
            span.set_attribute("input.length", len(input_data))
        elif isinstance(input_data, dict):
            span.set_attribute("input.data", json.dumps(input_data, default=str)[:1000])
    
    # Add custom attributes
    for key, value in additional_attributes.items():
        if value is not None:
            span.set_attribute(key, str(value))
    
    return span


def trace_agent_interaction(
    agent_name: str,
    operation: str,
    input_data: Optional[Union[str, Dict[str, Any]]] = None,
    **attributes
):
    """
    Create a span for agent interactions with conversation grouping.
    
    Args:
        agent_name: Name of the agent (e.g., 'ai_writer', 'ai_investigator')
        operation: Operation being performed (e.g., 'process_message', 'research')
        input_data: Input data for the operation
        **attributes: Additional span attributes
    """
    return create_conversation_span(
        f"agent.{agent_name}.{operation}",
        input_data=input_data,
        agent_name=agent_name,
        agent_type="llm_agent",
        operation_type="agent_interaction",
        **attributes
    )


def trace_tool_call(
    tool_name: str,
    input_params: Optional[Dict[str, Any]] = None,
    **attributes
):
    """
    Create a span for tool calls with conversation grouping.
    
    Args:
        tool_name: Name of the tool being called
        input_params: Tool input parameters
        **attributes: Additional span attributes
    """
    return create_conversation_span(
        f"tool.{tool_name}",
        input_data=input_params,
        tool_name=tool_name,
        tool_type="function_call",
        operation_type="tool_execution",
        **attributes
    )


def trace_conversation_turn(
    user_input: str,
    turn_number: Optional[int] = None,
    conversation_type: Optional[str] = None
):
    """
    Create a span for a complete conversation turn (user input + agent response).
    
    Args:
        user_input: The user's input message
        turn_number: Optional turn number in the conversation
        conversation_type: Type of conversation (e.g., 'cofacts', 'hackmd', 'general')
    """
    # Detect if this is a new conversation
    if detect_conversation_start(user_input):
        # Generate new conversation ID for new conversations
        new_conversation_id = generate_conversation_id()
        set_conversation_id(new_conversation_id)
        
        # Determine conversation type
        if not conversation_type:
            if "cofacts.tw" in user_input:
                conversation_type = "cofacts_factcheck"
            elif "hackmd.io" in user_input:
                conversation_type = "hackmd_analysis"
            else:
                conversation_type = "general_assistance"
    
    return create_conversation_span(
        "conversation.turn",
        input_data=user_input,
        turn_number=turn_number,
        conversation_type=conversation_type,
        operation_type="conversation_turn",
        user_input_length=len(user_input)
    )


def record_agent_output(span, output_data: Union[str, Dict[str, Any]], success: bool = True):
    """
    Record agent output data in the current span.
    
    Args:
        span: The active OpenTelemetry span
        output_data: The agent's output
        success: Whether the operation was successful
    """
    if isinstance(output_data, str):
        span.set_attribute("output.content", output_data[:1000])  # Truncate long outputs
        span.set_attribute("output.length", len(output_data))
    elif isinstance(output_data, dict):
        span.set_attribute("output.data", json.dumps(output_data, default=str)[:1000])
    
    span.set_attribute("operation.success", success)
    
    # Set span status
    if success:
        span.set_status(Status(StatusCode.OK))
    else:
        span.set_status(Status(StatusCode.ERROR))


def record_tool_result(span, result: Any, success: bool = True):
    """
    Record tool execution result in the current span.
    
    Args:
        span: The active OpenTelemetry span  
        result: The tool's result
        success: Whether the tool call was successful
    """
    if result is not None:
        result_str = str(result)
        span.set_attribute("tool.result_length", len(result_str))
        span.set_attribute("tool.result_preview", result_str[:500])  # Preview of result
        
        # Special handling for specific tool types
        if isinstance(result, dict):
            # Cofacts database results
            if "data" in result and isinstance(result["data"], dict):
                if "totalCount" in result["data"]:
                    span.set_attribute("cofacts.results_count", result["data"]["totalCount"])
            
            # Google search results metadata
            if "searchEntryPoint" in result:
                span.set_attribute("google_search.has_entry_point", True)
            if "candidates" in result:
                span.set_attribute("google_search.candidates_count", len(result["candidates"]))
    
    span.set_attribute("tool.success", success)
    
    # Set span status
    if success:
        span.set_status(Status(StatusCode.OK))
    else:
        span.set_status(Status(StatusCode.ERROR))


class ConversationTracer:
    """
    Context manager for tracing complete agent interactions within a conversation.
    
    Usage:
        with ConversationTracer("ai_writer", user_input="Check this message") as tracer:
            # Agent processing happens here
            result = agent.process(user_input)
            tracer.record_output(result)
    """
    
    def __init__(self, agent_name: str, operation: str = "process", **span_attributes):
        self.agent_name = agent_name
        self.operation = operation
        self.span_attributes = span_attributes
        self.span = None
    
    def __enter__(self):
        self.span = trace_agent_interaction(
            self.agent_name,
            self.operation,
            **self.span_attributes
        )
        self.span.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Record error in span
            self.span.set_attribute("error.type", exc_type.__name__)
            self.span.set_attribute("error.message", str(exc_val))
            self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
        
        self.span.__exit__(exc_type, exc_val, exc_tb)
    
    def record_output(self, output_data: Any, success: bool = True):
        """Record the agent's output in the span"""
        if self.span:
            record_agent_output(self.span, output_data, success)
    
    def record_intermediate_step(self, step_name: str, data: Any):
        """Record intermediate processing steps"""
        if self.span:
            self.span.set_attribute(f"step.{step_name}", str(data)[:200])


def get_conversation_context() -> Dict[str, str]:
    """
    Get current conversation context for logging or debugging.
    
    Returns:
        Dictionary with conversation context information
    """
    return {
        "conversation_id": conversation_id_var.get(),
        "has_active_conversation": conversation_id_var.get() is not None
    }