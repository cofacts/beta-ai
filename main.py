import os
import base64
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from langfuse import get_client

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    if OTEL_EXPORTER_OTLP_ENDPOINT:
        LANGFUSE_AUTH = base64.b64encode(
            f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()
        ).decode()
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

        provider = TracerProvider(resource=Resource.create({"service.name": "adk_langfuse_service"}))
        exporter = OTLPSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        print(f"Langfuse OpenTelemetry initialized. Endpoint: {OTEL_EXPORTER_OTLP_ENDPOINT}")
    else:
        print("OTEL_EXPORTER_OTLP_ENDPOINT not set. OpenTelemetry tracing will be handled by Langfuse SDK or other default providers.")

    # Instrument Google ADK
    GoogleADKInstrumentor().instrument()

    # Initialize Langfuse client
    # get_client() will use LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL from env
    langfuse = get_client()

    # Verify connection
    try:
        if langfuse.auth_check():
            print("Langfuse client is authenticated and ready!")
        else:
            print("Langfuse authentication failed. Please check your credentials and host.")
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
