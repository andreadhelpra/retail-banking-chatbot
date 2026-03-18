from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mistralai import Mistral

from app.agents import guardrails
from app.agents.action_agent import handle_action
from app.agents.faq_agent import handle_faq
from app.agents.supervisor import HANDOFF_MESSAGE, classify_intent
from app.config import MISTRAL_API_KEY
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    DebugInfo,
    SessionInfo,
    SessionState,
)
from app.services.mock_banking import MockBankingService
from app.services.retriever import FAQRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global state
sessions: dict[str, SessionState] = {}
mistral_client: Mistral | None = None
banking_service: MockBankingService | None = None
faq_retriever: FAQRetriever | None = None
other_customer_pii: list[str] = []

DEFAULT_CUSTOMER_ID = "cust_001"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mistral_client, banking_service, faq_retriever, other_customer_pii

    logger.info("Starting BNP Paribas AI Assistant...")

    mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    banking_service = MockBankingService()

    try:
        faq_retriever = FAQRetriever(mistral_client)
        logger.info("[STARTUP] FAQ retriever loaded successfully")
    except Exception as e:
        logger.error(f"[STARTUP] FAQ retriever failed to load: {e}")
        faq_retriever = None

    # Pre-compute PII list for other customers
    other_customer_pii = guardrails.get_other_customer_pii(banking_service, DEFAULT_CUSTOMER_ID)

    logger.info("[STARTUP] BNP Paribas AI Assistant ready")
    yield
    logger.info("Shutting down...")


app = FastAPI(title="BNP Paribas AI Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/session/init", response_model=SessionInfo)
async def init_session():
    global banking_service
    # Reload mock data so each session starts fresh
    banking_service = MockBankingService()

    session_id = str(uuid.uuid4())
    customer = banking_service.get_customer(DEFAULT_CUSTOMER_ID)

    sessions[session_id] = SessionState(
        session_id=session_id,
        customer_id=DEFAULT_CUSTOMER_ID,
    )

    customer_summary = {
        "name": customer["name"],
        "accounts": [
            {"label": a["label"], "type": a["type"], "last_iban": a["iban"][-4:]}
            for a in customer["accounts"]
        ],
        "cards": [
            {"label": c["label"], "type": c["type"], "last_four": c["last_four"], "status": c["status"]}
            for c in customer["cards"]
        ],
    }

    logger.info(f"[SESSION] New session: {session_id} for customer {customer['name']}")
    return SessionInfo(session_id=session_id, customer=customer_summary)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session = sessions.get(request.session_id)
    if not session:
        return ChatResponse(
            response="Session not found. Please initialize a new session.",
            debug=DebugInfo(agent="error", intent="error"),
        )

    customer_data = banking_service.get_customer(session.customer_id)
    message = request.message.strip()

    # --- Input guardrails ---
    input_check = guardrails.check_input(message)
    if not input_check["allowed"]:
        return ChatResponse(
            response=input_check["message"],
            debug=DebugInfo(
                agent="guardrails",
                intent=input_check["reason"],
            ),
        )

    # --- Check for pending action confirmation ---
    if session.pending_action:
        result = await handle_action(
            mistral_client, banking_service, message, session, customer_data, None
        )
        _add_to_history(session, message, result["response"])

        return ChatResponse(
            response=result["response"],
            debug=DebugInfo(
                agent="action",
                intent="action",
                tool_calls=result.get("tool_calls"),
            ),
        )

    # --- Supervisor: classify intent ---
    intent = await classify_intent(mistral_client, message, session, customer_data)

    # --- Route to appropriate agent ---
    if intent.intent == "clarify":
        response_text = intent.clarification_question or "Could you provide more details?"
        _add_to_history(session, message, response_text)
        return ChatResponse(
            response=response_text,
            debug=DebugInfo(
                agent="supervisor",
                intent="clarify",
                entities=intent.entities,
            ),
        )

    if intent.intent == "handoff":
        _add_to_history(session, message, HANDOFF_MESSAGE)
        return ChatResponse(
            response=HANDOFF_MESSAGE,
            debug=DebugInfo(
                agent="supervisor",
                intent="handoff",
                entities=intent.entities,
            ),
        )

    if intent.intent == "faq":
        if faq_retriever is None:
            response_text = "The FAQ service is temporarily unavailable. Please try again later."
            _add_to_history(session, message, response_text)
            return ChatResponse(
                response=response_text,
                debug=DebugInfo(agent="faq", intent="faq"),
            )

        result = await handle_faq(mistral_client, faq_retriever, message)
        response_text = result["response"]

        # Output guardrails
        output_check = guardrails.check_output(response_text, session.customer_id, other_customer_pii)
        if not output_check["allowed"]:
            response_text = output_check["message"]

        _add_to_history(session, message, response_text)

        return ChatResponse(
            response=response_text,
            debug=DebugInfo(
                agent="faq",
                intent="faq",
                entities=intent.entities,
                retrieved_chunks=result.get("retrieved_chunks"),
            ),
        )

    if intent.intent == "action":
        result = await handle_action(
            mistral_client, banking_service, message, session, customer_data, intent
        )
        response_text = result["response"]

        # Output guardrails
        output_check = guardrails.check_output(response_text, session.customer_id, other_customer_pii)
        if not output_check["allowed"]:
            response_text = output_check["message"]

        _add_to_history(session, message, response_text)

        return ChatResponse(
            response=response_text,
            debug=DebugInfo(
                agent="action",
                intent="action",
                entities=intent.entities,
                tool_calls=result.get("tool_calls"),
            ),
        )

    # Fallback
    return ChatResponse(
        response="I'm not sure how to help with that. Could you rephrase?",
        debug=DebugInfo(agent="supervisor", intent="unknown"),
    )


def _add_to_history(session: SessionState, user_msg: str, assistant_msg: str) -> None:
    """Add a turn to conversation history, capping at MAX_CONVERSATION_TURNS."""
    from app.config import MAX_CONVERSATION_TURNS

    session.conversation_history.append({"role": "user", "content": user_msg})
    session.conversation_history.append({"role": "assistant", "content": assistant_msg})

    # Cap history
    max_messages = MAX_CONVERSATION_TURNS * 2
    if len(session.conversation_history) > max_messages:
        session.conversation_history = session.conversation_history[-max_messages:]
