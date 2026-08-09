"""The LangGraph agent. YOU implement it — you've built this exact shape
before in the debugger project. (Step 6 of the path)

    START → agent → (tool call?) → tools → agent → ... → END

Definition of done: pytest tests/test_agent.py
"""

from __future__ import annotations
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (  # noqa: F401  (pre-imported for your implementation)
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from app.agent.state import AgentState  # noqa: F401  (pre-imported for your implementation)
from langgraph.prebuilt import ToolNode, tools_condition

MAX_INPUT_CHARS = 8_000
RECURSION_LIMIT = 15


def build_graph(llm: BaseChatModel, tools: list[BaseTool], system_prompt: str, checkpointer=None):
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        response = llm_with_tools.invoke([SystemMessage(system_prompt)] + list(state["messages"]))
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


async def run_turn(graph, text: str, psid: str) -> str:
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=text[:MAX_INPUT_CHARS])]},
        config={
            "configurable": {"thread_id": psid},
            "recursion_limit": RECURSION_LIMIT,
        },
    )
    content = result["messages"][-1].content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return str(content)