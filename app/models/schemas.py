from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str
    message: str


class IntentClassification(BaseModel):
    intent: Literal["faq", "action", "handoff", "clarify"]
    entities: dict[str, Any] = Field(default_factory=dict)
    clarification_question: str | None = None


class PendingAction(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    confirmation_message: str


class DebugInfo(BaseModel):
    agent: str
    intent: str
    entities: dict[str, Any] = Field(default_factory=dict)
    retrieved_chunks: list[str] | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatResponse(BaseModel):
    response: str
    debug: DebugInfo


class SessionInfo(BaseModel):
    session_id: str
    customer: dict[str, Any]


class SessionState(BaseModel):
    session_id: str
    customer_id: str
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    pending_action: PendingAction | None = None
