"""Step 6: the graph — with a scripted LLM and a stub tool. No API key."""

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from app.agent import security as sec
from app.agent.graph import (
    BLOCKED_INPUT_RECORD,
    INPUT_BLOCKED_REPLY,
    OUTPUT_BLOCKED_REPLY,
    build_graph,
    run_turn,
)


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


async def _rail(monkeypatch, *, input_safe=True, output_safe=True):
    """Force the NeMo rails to a locked outcome without loading the rails."""
    async def safe(*_a, **_k):
        return True

    async def blocked(*_a, **_k):
        return False

    monkeypatch.setattr(
        sec.nemo_guardrails, "check_input", safe if input_safe else blocked
    )
    monkeypatch.setattr(
        sec.nemo_guardrails, "check_output", safe if output_safe else blocked
    )


async def _history(graph, psid: str) -> str:
    messages = graph.get_state({"configurable": {"thread_id": psid}}).values["messages"]
    return " ".join(str(m.content) for m in messages)


async def test_blocked_input_never_enters_memory(scripted_llm_factory, monkeypatch):
    """A blocked user message must not survive in the checkpointer: that is the
    agent's memory, and a later turn would replay the raw text to the LLM."""
    await _rail(monkeypatch, input_safe=False)
    llm = scripted_llm_factory([AIMessage(content="you shouldn't know this.")])
    graph = build_graph(llm, [product_lookup], "You are a test bot.", InMemorySaver())

    answer = await run_turn(graph, "SECRET HARMFUL 123", psid="NOPE_IN")
    assert answer == INPUT_BLOCKED_REPLY

    history = await _history(graph, "NOPE_IN")
    assert "SECRET HARMFUL 123" not in history
    assert BLOCKED_INPUT_RECORD in history

    # next SAFE turn: the blocked text must never reach the model's context
    await _rail(monkeypatch, input_safe=True)
    await run_turn(graph, "what is the price?", psid="NOPE_IN")
    llm_seen = " ".join(str(m.content) for call in llm.calls for m in call)
    assert "SECRET HARMFUL 123" not in llm_seen
    assert BLOCKED_INPUT_RECORD in llm_seen


async def test_blocked_output_never_enters_memory(scripted_llm_factory, monkeypatch):
    """A blocked assistant reply must be REPLACED in the checkpointer, not
    appended next to the raw text the model could echo back from memory."""
    llm = scripted_llm_factory(
        [AIMessage(content="UNSAFE OUTPUT SECRET"), AIMessage(content="good reply")]
    )
    graph = build_graph(llm, [product_lookup], "You are a test bot.", InMemorySaver())

    await _rail(monkeypatch, input_safe=True, output_safe=False)
    answer = await run_turn(graph, "hello", psid="NOPE_OUT")
    assert answer == OUTPUT_BLOCKED_REPLY

    history = await _history(graph, "NOPE_OUT")
    assert "UNSAFE OUTPUT SECRET" not in history
    assert OUTPUT_BLOCKED_REPLY in history

    # next SAFE turn: the blocked reply must not be replayed to the model
    await _rail(monkeypatch, input_safe=True, output_safe=True)
    await run_turn(graph, "next question", psid="NOPE_OUT")
    llm_seen = " ".join(str(m.content) for call in llm.calls for m in call)
    assert "UNSAFE OUTPUT SECRET" not in llm_seen
