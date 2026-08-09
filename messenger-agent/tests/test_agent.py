"""Step 6: the graph — with a scripted LLM and a stub tool. No API key."""

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import build_graph, run_turn


@tool
def product_lookup(query: str) -> str:
    """Look up a product's exact price and stock."""
    return "Veste Rouge Classic — 3500 DA — in stock: 4"


def scripted(factory):
    return factory(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "product_lookup",
                        "args": {"query": "veste rouge"},
                        "id": "c1",
                    }
                ],
            ),
            AIMessage(content="La Veste Rouge Classic coûte 3500 DA, il en reste 4."),
        ]
    )


async def test_react_loop_uses_tool_then_answers(scripted_llm_factory):
    graph = build_graph(
        scripted(scripted_llm_factory), [product_lookup], "You are a test bot.", InMemorySaver()
    )
    answer = await run_turn(graph, "combien coute la veste rouge?", psid="PSID_1")
    assert "3500" in answer


async def test_tool_result_actually_flowed_through_graph(scripted_llm_factory):
    graph = build_graph(
        scripted(scripted_llm_factory), [product_lookup], "You are a test bot.", InMemorySaver()
    )
    await run_turn(graph, "prix de la veste?", psid="PSID_2")
    state = graph.get_state({"configurable": {"thread_id": "PSID_2"}})
    assert any(isinstance(m, ToolMessage) and "3500" in str(m.content) for m in state.values["messages"])


async def test_memory_per_customer_thread(scripted_llm_factory):
    """thread_id = psid: customer A's history grows; customer B starts clean."""
    graph = build_graph(
        scripted(scripted_llm_factory), [product_lookup], "You are a test bot.", InMemorySaver()
    )
    await run_turn(graph, "question 1", psid="A")
    await run_turn(graph, "question 2", psid="A")
    a = len(graph.get_state({"configurable": {"thread_id": "A"}}).values["messages"])
    b_state = graph.get_state({"configurable": {"thread_id": "B"}}).values
    assert a >= 6  # two full turns accumulated
    assert len(b_state.get("messages", [])) == 0  # B untouched


async def test_system_prompt_prepended(scripted_llm_factory):
    llm = scripted(scripted_llm_factory)
    graph = build_graph(llm, [product_lookup], "GUARDRAILS HERE", InMemorySaver())
    await run_turn(graph, "hello", psid="C")
    first_call_messages = llm.calls[0]
    assert first_call_messages[0].type == "system"
    assert "GUARDRAILS" in first_call_messages[0].content
