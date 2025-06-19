from phoenix.evals import ClassificationTemplate

AGENT_HANDOFF_TEMPLATE = ClassificationTemplate(
    rails=["correct_handoff", "incorrect_handoff"],
    template="""
You are evaluating the correctness of agent handoffs in a travel concierge system.

User Query: {query}
Agent Response: {agent_response}
Expected Agent Transfers: {expected_agent_transfers}
Actual Agent Transfers: {actual_agent_transfers}

Evaluate whether the agent correctly transferred to the appropriate sub-agents based on the user's query and expected behavior.

Classification Guidelines:
- correct_handoff: The agent transferred to the correct sub-agent(s) as expected, OR correctly chose not to transfer when no transfer was needed
- incorrect_handoff: The agent failed to transfer when needed, transferred to wrong agent(s), or made unnecessary transfers

Classify this agent handoff behavior:
"""
)


TOOL_USAGE_TEMPLATE = ClassificationTemplate(
    rails=["correct_tools", "incorrect_tools"],
    template="""
You are evaluating the correctness of tool usage within travel concierge agents.

User Query: {query}
Agent Response: {agent_response}
Expected Tools Used: {expected_other_tools}
Actual Tools Used: {actual_tool_calls}

Evaluate whether the agent used the appropriate tools based on the user's query and expected behavior.

Classification Guidelines:
- correct_tools: The agent used the expected tools appropriately, OR correctly chose not to use tools when none were needed
- incorrect_tools: The agent failed to use expected tools, used wrong tools, or used tools unnecessarily

Classify this tool usage behavior:
"""
)


RESPONSE_QUALITY_TEMPLATE = ClassificationTemplate(
    rails=["good_response", "poor_response"],
    template="""
You are evaluating the quality of travel concierge agent responses.

User Query: {query}
Agent Response: {agent_response}
Expected Response: {expected_response}

Evaluate the quality of the agent's response based on helpfulness, relevance, and appropriateness for a travel concierge.

Classification Guidelines:
- good_response: The response is helpful, relevant, and addresses the user's query appropriately for travel assistance
- poor_response: The response is unhelpful, irrelevant, incomplete, or inappropriate for travel assistance

Consider the following factors:
- Does the response directly address the user's query?
- Is the response helpful for travel planning/assistance?
- Is the tone appropriate and professional?
- Does the response demonstrate travel domain knowledge?

Classify this response quality:
"""
)