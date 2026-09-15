"""Agent state — provided complete (a declaration you already know well).

add_messages accumulates within ONE run. Memory ACROSS runs comes from the
checkpointer + thread_id (= the customer's PSID). Same lesson as the
state lab from the debugger project.
"""

from __future__ import annotations

from typing import Annotated, NotRequired, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    is_safe: NotRequired[bool]
    # Set by the input guardrail node when it blocks a message. Routing reads
    # this flag instead of matching reply text, which is how a blocked input
    # used to slip through to the model.
    input_blocked: NotRequired[bool]
