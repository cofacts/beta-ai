import os
import base64

import uvicorn
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app

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


# Get the directory where main.py is located
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Example allowed origins for CORS
ALLOWED_ORIGINS = []
# Set web=True if you intend to serve a web interface, False otherwise
SERVE_WEB_INTERFACE = True

# Call the function to get the FastAPI app instance
# Ensure the agent directory name ('capital_agent') matches your agent folder
app: FastAPI = get_fast_api_app(
  agent_dir=AGENT_DIR,
  allow_origins=ALLOWED_ORIGINS,
  web=SERVE_WEB_INTERFACE,
)

if __name__ == "__main__":
  # Use the PORT environment variable provided by Cloud Run, defaulting to 8080
  uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
