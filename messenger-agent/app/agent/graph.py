"""The LangGraph agent.

    START → agent → (tool call?) → tools → agent → ... → END

Definition of done: pytest tests/test_agent.py
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from app.util.retry import with_retry
from app.agent.state import AgentState
from app.agent.security import (
    nemo_guardrails,
    pii_input_middleware,
    pii_output_middleware,
)
from app.config import get_settings

MAX_INPUT_CHARS = 8_000
RECURSION_LIMIT = 15


def build_graph(llm: BaseChatModel, tools: list[BaseTool], system_prompt: str, checkpointer=None):
    llm_with_tools = llm.bind_tools(tools)
    settings = get_settings()

    async def agent_node(state: AgentState):
        trimmed_messages = trim_messages(
            state["messages"],
            max_tokens=settings.history_max_messages,
            strategy="last",
            token_counter=len,
            start_on="human",
            include_system=False,
            allow_partial=False,
        )
        messages = [SystemMessage(system_prompt)] + list(trimmed_messages)
        response = await with_retry(
            lambda: llm_with_tools.ainvoke(messages),
            attempts=3,
            budget=12.0,
            label="gemini",
        )
        response = response.model_copy()
        response.id = None
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


async def run_turn(graph, text: str, psid: str, thread_id: str | None = None) -> str:
    tid = thread_id or psid
    user_text = text[:MAX_INPUT_CHARS]

    # 1. PII: mask credit cards in user input (LangChain prebuilt middleware)
    if hasattr(pii_input_middleware, "mask"):
        user_text = pii_input_middleware.mask(user_text)

    # 2. NeMo Guardrails INPUT rail
    is_safe = await nemo_guardrails.check_input(user_text)
    if not is_safe:
        return "Sma7lna, ma n9derch njawbek 3la had l'demande."

    # 3. Graph Invocation
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=user_text)]},
        config={
            "configurable": {"thread_id": tid},
            "recursion_limit": RECURSION_LIMIT,
        },
    )

    content = result["messages"][-1].content
    if isinstance(content, list):
        reply = "".join(block.get("text", "") for block in content if isinstance(block, dict))
    else:
        reply = str(content)

    # 4. NeMo Guardrails OUTPUT rail — validate the reply before it reaches the user
    output_safe = await nemo_guardrails.check_output(reply, user_text=user_text)
    if not output_safe:
        return "Sma7lna, ma n9derch njawbek 3la had l'demande."

    # 5. PII: mask any credit cards that slipped into the model's reply
    if hasattr(pii_output_middleware, "mask"):
        reply = pii_output_middleware.mask(reply)

    return reply