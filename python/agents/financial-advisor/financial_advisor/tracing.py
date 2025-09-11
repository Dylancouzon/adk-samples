import os
from phoenix.otel import register
from opentelemetry import trace
from typing import Optional
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
import warnings

from dotenv import load_dotenv
load_dotenv()

def instrument_adk_with_phoenix() -> Optional[trace.Tracer]:
    """Instrument the ADK with Arize Phoenix Cloud."""
    
    # Phoenix Cloud requires both API key and collector endpoint
    api_key = os.getenv("PHOENIX_API_KEY")
    collector_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    project_name = os.getenv("PHOENIX_PROJECT_NAME", "adk-financial-advisor")
    
    if api_key is None:
        warnings.warn("PHOENIX_API_KEY is not set - required for Phoenix Cloud")
        return None
    
    if collector_endpoint is None:
        warnings.warn("PHOENIX_COLLECTOR_ENDPOINT is not set - required for Phoenix Cloud")
        return None
    
    # Set environment variables for phoenix.otel.register to pick up
    os.environ["PHOENIX_API_KEY"] = api_key
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = collector_endpoint
    
    tracer_provider = register(
        project_name=project_name,
        auto_instrument=True,  # Enable automatic instrumentation
    )

    GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)

    return tracer_provider.get_tracer(__name__)