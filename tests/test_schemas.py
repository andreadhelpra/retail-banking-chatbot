from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    DebugInfo,
    IntentClassification,
    PendingAction,
)


def test_chat_request_valid():
    req = ChatRequest(session_id="sess_001", message="Hello")
    assert req.session_id == "sess_001"
    assert req.message == "Hello"


def test_intent_classification():
    intent = IntentClassification(
        intent="faq",
        confidence=0.92,
        entities={},
        clarification_question=None,
    )
    assert intent.intent == "faq"
    assert intent.confidence == 0.92


def test_debug_info_defaults():
    debug = DebugInfo(agent="faq", intent="faq", confidence=0.9)
    assert debug.retrieved_chunks is None
    assert debug.tool_calls is None
    assert debug.entities == {}


def test_chat_response():
    resp = ChatResponse(
        response="Hello!",
        debug=DebugInfo(agent="supervisor", intent="clarify", confidence=0.7),
    )
    assert resp.response == "Hello!"
    assert resp.debug.agent == "supervisor"


def test_pending_action():
    action = PendingAction(
        tool_name="lock_card",
        arguments={"card_id": "card_002", "lock_type": "temporary"},
        confirmation_message="Lock Visa ****4821 temporarily?",
    )
    assert action.tool_name == "lock_card"
