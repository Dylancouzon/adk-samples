import os
from arize.otel import register
from opentelemetry import trace
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from dotenv import load_dotenv
load_dotenv()

def instrument_adk_with_arize() -> trace.Tracer:
    """Instrument the ADK with Arize."""
    
    if os.getenv("ARIZE_SPACE_ID") is None:
        raise ValueError("ARIZE_SPACE_ID is not set")
    if os.getenv("ARIZE_API_KEY") is None:
        raise ValueError("ARIZE_API_KEY is not set")
    
    tracer_provider = register(
        space_id = os.getenv("ARIZE_SPACE_ID"),
        api_key = os.getenv("ARIZE_API_KEY"),
        project_name = os.getenv("ARIZE_PROJECT_NAME", "adk-rag-agent"),
    )

    GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)

    return tracer_provider.get_tracer(__name__)