"""Upload Financial Advisor prompts to Arize Phoenix.

Usage:
  - Ensure environment variables are set:
      PHOENIX_API_KEY
      PHOENIX_CLIENT_HEADERS (e.g., "api_key=..."), if PHOENIX_API_KEY is not set
      PHOENIX_BASE_URL or PHOENIX_COLLECTOR_ENDPOINT
  - Then run:
      python financial_advisor/scripts/upload_prompts.py
"""

from __future__ import annotations

import os
import warnings
import dotenv

from phoenix.client import Client  # type: ignore
from phoenix.client.types import PromptVersion  # type: ignore
from pathlib import Path
import httpx  # type: ignore


MODEL = "gemini-2.5-pro"

# Load environment variables from .env file
dotenv.load_dotenv()


def _bridge_env() -> None:
    api_key = os.getenv("PHOENIX_API_KEY")
    if api_key and not os.getenv("PHOENIX_CLIENT_HEADERS"):
        os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={api_key}"


def _ensure_prompt(client: Client, name: str, description: str, content: str) -> None:
    version = PromptVersion(
        [{"role": "user", "content": content}],
        model_name=MODEL,
        model_provider="GOOGLE",
    )
    try:
        created = client.prompts.create(
            name=name,
            prompt_description=description,
            version=version,
        )
        print(f"Created: {name} (version_id={getattr(created, 'id', 'unknown')})")
        
        # Add production tag
        client.prompts.tags.create(
            prompt_version_id=created.id,
            name="production",
            description="Ready for production environment"
        )
        
    except httpx.HTTPStatusError as e:
        if e.response is not None and e.response.status_code == 409:
            # Already exists -> create new version to refresh provider
            updated = client.prompts.create(
                name=name,
                prompt_description=description,
                version=version,
            )
            print(f"Updated: {name} (new_version_id={getattr(updated, 'id', 'unknown')})")
            
            # Add production tag
            client.prompts.tags.create(
                prompt_version_id=updated.id,
                name="production",
                description="Ready for production environment"
            )
            return
        body = e.response.text if e.response is not None else "<no body>"
        raise RuntimeError(f"Failed to create prompt '{name}': {e} Body: {body}")


def _load_constant_from_file(file_path: Path, constant_name: str) -> str:
    ns: dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    # Execute in isolated namespace to capture constants
    exec(compile(code, str(file_path), "exec"), ns, ns)
    value = ns.get(constant_name)
    if not isinstance(value, str):
        raise RuntimeError(f"Constant {constant_name} not found in {file_path}")
    return value


def main() -> int:
    _bridge_env()
    api_key = os.getenv("PHOENIX_API_KEY")
    collector = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")

    try:
        if api_key and collector:
            client = Client(base_url=collector, headers={"api_key": api_key})
        else:
            client = Client()
    except Exception as e:
        warnings.warn(
            "Failed to initialize Phoenix Client. Set PHOENIX_CLIENT_HEADERS and "
            "PHOENIX_BASE_URL/PHOENIX_COLLECTOR_ENDPOINT."
        )
        print(str(e))
        return 1

    # Resolve file paths without importing package top-level to avoid side effects
    scripts_dir = Path(__file__).resolve().parent
    fa_dir = scripts_dir.parent

    coordinator_path = fa_dir / "prompt.py"
    da_path = fa_dir / "sub_agents" / "data_analyst" / "prompt.py"
    ta_path = fa_dir / "sub_agents" / "trading_analyst" / "prompt.py"
    ea_path = fa_dir / "sub_agents" / "execution_analyst" / "prompt.py"
    ra_path = fa_dir / "sub_agents" / "risk_analyst" / "prompt.py"

    coordinator_content = _load_constant_from_file(coordinator_path, "FINANCIAL_COORDINATOR_PROMPT")
    da_content = _load_constant_from_file(da_path, "DATA_ANALYST_PROMPT")
    ta_content = _load_constant_from_file(ta_path, "TRADING_ANALYST_PROMPT")
    ea_content = _load_constant_from_file(ea_path, "EXECUTION_ANALYST_PROMPT")
    ra_content = _load_constant_from_file(ra_path, "RISK_ANALYST_PROMPT")

    _ensure_prompt(
        client,
        name="financial-advisor-financial-coordinator",
        description="Root coordinator prompt for the Financial Advisor agent",
        content=coordinator_content,
    )
    _ensure_prompt(
        client,
        name="financial-advisor-data-analyst",
        description="Data analyst prompt for market data collection and synthesis",
        content=da_content,
    )
    _ensure_prompt(
        client,
        name="financial-advisor-trading-analyst",
        description="Trading analyst prompt for proposing trading strategies",
        content=ta_content,
    )
    _ensure_prompt(
        client,
        name="financial-advisor-execution-analyst",
        description="Execution analyst prompt for building execution plans",
        content=ea_content,
    )
    _ensure_prompt(
        client,
        name="financial-advisor-risk-analyst",
        description="Risk analyst prompt for final risk evaluation",
        content=ra_content,
    )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


