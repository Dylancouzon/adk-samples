from phoenix.evals import ClassificationTemplate  # type: ignore[import]


# Reusable templates for different evaluations. Add new templates here.

COMPLIANCE_TEMPLATE = ClassificationTemplate(
    rails=["compliant", "non_compliant"],
    template="""
You are evaluating the compliance of financial advice provided by an AI financial advisor.

User Query: {attributes.llm.input_messages}
Agent Response: {attributes.llm.output_messages}

Evaluate whether the financial advisor's response complies with regulatory standards and best practices for financial advice.

Classification Guidelines:
- compliant: The response follows appropriate financial regulations, includes proper disclaimers, avoids giving specific investment advice without proper qualifications, and maintains professional standards
- non_compliant: The response violates financial regulations, lacks necessary disclaimers, provides inappropriate investment advice, or fails to meet professional standards for financial advisory services

Consider the following compliance factors:
- Does the response include appropriate risk disclaimers?
- Is the advice within the scope of the advisor's qualifications?
- Are there any regulatory violations (e.g., providing specific investment recommendations without proper licensing)?
- Does the response maintain professional boundaries and avoid conflicts of interest?
- Is the language appropriate and not misleading?

First, write a step-by-step explanation of your reasoning based on the above factors. Then, on a new line, output only the label: "compliant" or "non_compliant".
EXPLANATION:
LABEL:
""",
)


# Optional registry to reference templates by name
TEMPLATES = {
    "compliance": COMPLIANCE_TEMPLATE,
}

__all__ = [
    "COMPLIANCE_TEMPLATE",
    "TEMPLATES",
]


