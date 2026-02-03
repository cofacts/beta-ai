import os
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from langfuse import get_client

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

from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app

# Get the directory where main.py is located
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Example allowed origins for CORS
ALLOWED_ORIGINS = []
# Set web=True if you intend to serve a web interface, False otherwise
SERVE_WEB_INTERFACE = True

# Call the function to get the FastAPI app instance
# Ensure the agent directory name ('capital_agent') matches your agent folder
app: FastAPI = get_fast_api_app(
  agents_dir=AGENT_DIR,
  allow_origins=ALLOWED_ORIGINS,
  web=SERVE_WEB_INTERFACE,
)

if __name__ == "__main__":
  import uvicorn
  # Use the PORT environment variable provided by Cloud Run, defaulting to 8080
  uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
