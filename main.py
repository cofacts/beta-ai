import os
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from langfuse import get_client
from fastapi import FastAPI
from ag_ui_adk import ADKAgent, add_adk_fastapi_endpoint
from cofacts_ai.agent import ai_writer

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")

if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    # Set default host and service name for OpenTelemetry/Langfuse if not provided
    if not os.getenv("LANGFUSE_BASE_URL"):
        os.environ["LANGFUSE_BASE_URL"] = "https://langfuse.cofacts.tw"
    if not os.getenv("OTEL_SERVICE_NAME"):
        os.environ["OTEL_SERVICE_NAME"] = "adk_langfuse_service"

    # Initialize Langfuse client
    # get_client() will use LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL from env
    # It also initializes the OpenTelemetry TracerProvider with a Langfuse SpanProcessor
    langfuse = get_client()

    # Instrument Google ADK
    GoogleADKInstrumentor().instrument()

    # Verify connection
    host = os.environ["LANGFUSE_BASE_URL"]
    try:
        if langfuse.auth_check():
            print(f"Langfuse client is authenticated and ready! (Host: {host})")
        else:
            print(f"Langfuse authentication failed. Please check your credentials and host ({host}).")
    except Exception as e:
        print(f"Error checking Langfuse authentication: {e}")
else:
    print("Langfuse credentials not set. Tracing and Langfuse client will be disabled.")

# Initialize FastAPI app
app = FastAPI(title="Cofacts AI CopilotKit Agent")

# Wrap the ADK agent with ADK Middleware
# We use ai_writer as the main entry agent
adk_agent = ADKAgent(
    adk_agent=ai_writer,
    app_name="cofacts_ai",
    user_id="demo_user",
    session_timeout_seconds=3600,
    use_in_memory_services=True
)

# Add the ADK endpoint to the FastAPI app
# This exposes the agent via AG-UI protocol at the root path
add_adk_fastapi_endpoint(app, adk_agent, path="/")

if __name__ == "__main__":
    import uvicorn
    # Use the PORT environment variable provided by Cloud Run, defaulting to 8080
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
