import os
import base64
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY and OTEL_EXPORTER_OTLP_ENDPOINT:
    LANGFUSE_AUTH = base64.b64encode(
        f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()
    ).decode()
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"
    # OTEL_EXPORTER_OTLP_ENDPOINT is now also read from env and checked

    provider = TracerProvider(resource=Resource.create({"service.name": "adk_langfuse_service"}))
    exporter = OTLPSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("adk_application_tracer")
    print(f"Langfuse OpenTelemetry initialized. Endpoint: {OTEL_EXPORTER_OTLP_ENDPOINT}")
else:
    print(f"Langfuse OpenTelemetry tracing will be disabled.")

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from google.adk.cli.fast_api import get_fast_api_app
from langfuse_tracing import (
    set_conversation_id,
    get_or_create_conversation_id,
    detect_conversation_start,
    create_conversation_span
)

# Get the directory where main.py is located
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Example allowed origins for CORS
ALLOWED_ORIGINS = []
# Set web=True if you intend to serve a web interface, False otherwise
SERVE_WEB_INTERFACE = True

class ConversationTracingMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically track conversations for Langfuse grouping"""
    
    async def dispatch(self, request: Request, call_next):
        # Extract conversation ID from headers or generate new one
        conversation_id = request.headers.get("x-conversation-id")
        
        # For POST requests to agent endpoints, check if this starts a new conversation
        if request.method == "POST" and "agent" in str(request.url.path):
            try:
                # Try to read request body to detect conversation start
                body = await request.body()
                if body:
                    import json
                    try:
                        body_data = json.loads(body.decode())
                        user_input = body_data.get("message", "")
                        
                        # If this looks like a new conversation or no existing ID, create new one
                        if detect_conversation_start(user_input) or not conversation_id:
                            conversation_id = get_or_create_conversation_id()
                        elif conversation_id:
                            set_conversation_id(conversation_id)
                            
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
                
                # Recreate request with original body
                async def receive():
                    return {"type": "http.request", "body": body, "more_body": False}
                
                request._receive = receive
                
            except Exception:
                # Fallback: just ensure we have a conversation ID
                if not conversation_id:
                    conversation_id = get_or_create_conversation_id()
                else:
                    set_conversation_id(conversation_id)
        else:
            # For other requests, use existing ID or create new one
            if not conversation_id:
                conversation_id = get_or_create_conversation_id()
            else:
                set_conversation_id(conversation_id)
        
        # Create span for the HTTP request
        with create_conversation_span(
            "http.request",
            http_method=request.method,
            http_path=request.url.path,
            conversation_id=conversation_id
        ) as span:
            try:
                response = await call_next(request)
                
                # Add conversation ID to response headers for client to track
                response.headers["x-conversation-id"] = conversation_id
                span.set_attribute("http.status_code", response.status_code)
                
                return response
            except Exception as e:
                span.set_attribute("http.error", str(e))
                raise


# Call the function to get the FastAPI app instance
app: FastAPI = get_fast_api_app(
  agents_dir=AGENT_DIR,
  allow_origins=ALLOWED_ORIGINS,
  web=SERVE_WEB_INTERFACE,
)

# Add conversation tracing middleware
app.add_middleware(ConversationTracingMiddleware)

if __name__ == "__main__":
  import uvicorn
  # Use the PORT environment variable provided by Cloud Run, defaulting to 8080
  uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
