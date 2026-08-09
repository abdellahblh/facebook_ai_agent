"""Agent state — provided complete (a declaration you already know well).

add_messages accumulates within ONE run. Memory ACROSS runs comes from the
checkpointer + thread_id (= the customer's PSID). Same lesson as the
state lab from the debugger project.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
