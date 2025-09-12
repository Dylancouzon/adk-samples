import os
from typing import Dict, List, Optional

import dotenv
import vertexai
import phoenix as px  # type: ignore[import]
from phoenix.trace import SpanEvaluations  # type: ignore[import]
from phoenix.evals import llm_classify, GeminiModel  # type: ignore[import]

from . import templates


def _get_llm_eval_dataframe(project_name: str):
    """
    Return a dataframe of LLM spans with the columns required for evaluations.
    Index is set to the span_id so results can be logged back to Phoenix.
    """
    spans_df = px.Client().get_spans_dataframe(project_name=project_name)
    llm_spans = spans_df[spans_df["span_kind"] == "LLM"]
    eval_df = llm_spans[[
        "context.span_id",
        "attributes.llm.input_messages",
        "attributes.llm.output_messages",
    ]].copy()
    eval_df.set_index("context.span_id", inplace=True)
    return eval_df


def run_compliance_eval(
    project_name: str,
    model_name: str = "gemini-2.5-flash",
    location: str = "us-central1",
):
    """
    Classify each LLM span as compliant or non_compliant and log evaluations.
    """
    google_cloud_project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not google_cloud_project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set")

    vertexai.init(project=google_cloud_project, location=location)
    phoenix_model = GeminiModel(
        model=model_name,
        project=google_cloud_project,
        location=location,
    )

    eval_df = _get_llm_eval_dataframe(project_name)

    classifications = llm_classify(
        dataframe=eval_df,
        template=templates.COMPLIANCE_TEMPLATE,
        model=phoenix_model,
        rails=templates.COMPLIANCE_TEMPLATE.rails,
        provide_explanation=True,
    )

    classifications["score"] = classifications["label"].apply(
        lambda x: 1 if x == "compliant" else 0
    )

    px.Client().log_evaluations(
        SpanEvaluations(eval_name="compliance", dataframe=classifications)
    )

    return classifications


# Registry of available eval runners. Add new evals here.
from typing import Callable

EVAL_RUNNERS: Dict[str, Callable[..., object]] = {
    "compliance": run_compliance_eval,
}


def run(
    evals: Optional[List[str]] = None,
    project_name: str = "adk-financial-advisor",
):
    """
    Run selected evaluations (or all if none specified) and return results.
    """
    selected = evals or list(EVAL_RUNNERS.keys())
    results: Dict[str, object] = {}
    for name in selected:
        runner = EVAL_RUNNERS.get(name)
        if not runner:
            continue
        results[name] = runner(project_name=project_name)
    return results


if __name__ == "__main__":
    dotenv.load_dotenv()
    evals_env = os.getenv("EVALS", "").strip()
    selected_evals = [e.strip() for e in evals_env.split(",") if e.strip()] or None
    project = os.getenv("PHOENIX_PROJECT_NAME", "adk-financial-advisor")
    run(evals=selected_evals, project_name=project)


