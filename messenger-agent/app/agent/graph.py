"""The LangGraph agent.

    START → agent → (tool call?) → tools → agent → ... → END

Definition of done: pytest tests/test_agent.py
"""

from __future__ import annotations

import logging
import uuid

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler
from typing import Optional
langfuse_handler = CallbackHandler()
from app.agent.guardrail import (
    mask_input_pii,
    mask_output_pii,
)
from app.agent.state import AgentState
from app.config import get_settings
from app.util.retry import with_retry

logger = logging.getLogger(__name__)

MAX_INPUT_CHARS = 8_000
RECURSION_LIMIT = 15

# Placeholder that REPLACES a blocked message INSIDE the checkpointer. The raw
# text must never be persisted: it is the agent's memory and would otherwise be
# fed back to the LLM on the next turn of the same conversation.


def build_graph(llm: BaseChatModel, tools: list[BaseTool], system_prompt: str, checkpointer=None):
    llm_with_tools = llm.bind_tools(tools)
    settings = get_settings()

    async def agent_node(state: AgentState):
        def count_tokens_approximately(messages, chars_per_token=4.0):
            
            total_chars = sum(len(str(getattr(m, "content", m))) for m in messages)
            return int(total_chars / chars_per_token)


        trimmed_messages = trim_messages(
            state["messages"],
            max_tokens=settings.history_max_messages * 500,
            strategy="last",
            token_counter=count_tokens_approximately,
            start_on="human",
            include_system=False,
            allow_partial=False,
        )

        # Gemini API requires non-empty content for messages without tool calls
        sanitized_messages = []
        for msg in trimmed_messages:
            has_tools = bool(getattr(msg, "tool_calls", None))
            if not has_tools and (msg.content is None or str(msg.content).strip() == ""):
                msg = msg.model_copy(update={"content": " "})
            sanitized_messages.append(msg)

        messages = [SystemMessage(system_prompt)] + sanitized_messages
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


async def run_turn(
    graph,
    text: str,
    psid: str,
    thread_id: str | None = None,
    config_context: dict | None = None,
) -> str:
    tid = thread_id or psid
    user_text = text[:MAX_INPUT_CHARS]

    # 1. PII: mask credit cards in user input.
    user_text = mask_input_pii(user_text)

    configurable = {"thread_id": tid, "psid": psid}
    if config_context:
        configurable.update(config_context)

    # The graph owns NeMo input/output checks, so a message is not billed twice.
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=user_text)]},
        config={
            "configurable": configurable,
            "recursion_limit": RECURSION_LIMIT,
            "callbacks": [langfuse_handler]
        },
    )

    content = result["messages"][-1].content
    if isinstance(content, list):
        reply = "".join(block.get("text", "") for block in content if isinstance(block, dict))
    else:
        reply = str(content)

    # Output rails ran in the graph's final node. Mask PII before delivery.
    reply = mask_output_pii(reply)

    return reply

