from __future__ import annotations

import json
import logging
from typing import Any

from app.config import CONFIDENCE_THRESHOLD, MISTRAL_LARGE_MODEL
from app.models.schemas import IntentClassification, SessionState

logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM_PROMPT = """You are the supervisor agent for BNP Paribas retail banking assistant.
Your job is to classify the user's intent and extract relevant entities.

You must respond with a JSON object containing:
- "intent": one of "faq", "action", "handoff", "clarify"
- "confidence": a float between 0 and 1
- "entities": an object with extracted entities (card_id, account_id, account_type, card_type, lock_type, days, etc.)
- "clarification_question": a string (only if intent is "clarify")

Intent definitions:
- "faq": User asks a general banking question (hours, rates, fees, how-to). No account-specific action needed.
- "action": User wants to perform an operation on their account (check balance, view transactions, lock a card).
- "handoff": User has a complaint, dispute, or complex issue that requires a human agent.
- "clarify": The user's request is ambiguous and you need more information to route correctly.

The user is authenticated as: {customer_name}
Their accounts: {accounts_summary}
Their cards: {cards_summary}

Rules:
- If the user mentions "card" but has multiple cards without specifying which one, set intent to "clarify" and ask which card.
- If the user mentions "account" but has multiple accounts without specifying which, and the action requires a specific account, set intent to "clarify".
- For balance checks, if the user says "my balance" without specifying, default to the checking account.
- If the user says they lost or had their card stolen, set intent to "action" with entities including lock_type ("temporary" for lost, "permanent" for stolen).
- If confidence is below {confidence_threshold}, set intent to "clarify".
- For "block my card" without specifying which card, set intent to "clarify" and ask which card they want to block.
"""


async def classify_intent(
    mistral_client,
    message: str,
    session: SessionState,
    customer_data: dict[str, Any],
) -> IntentClassification:
    """Classify user intent using Mistral Large with structured JSON output."""

    accounts_summary = ", ".join(
        f"{a['label']} ({a['type']}, ****{a['iban'][-3:]})" for a in customer_data.get("accounts", [])
    )
    cards_summary = ", ".join(
        f"{c['label']} (****{c['last_four']}, {c['type']}, status: {c['status']})"
        for c in customer_data.get("cards", [])
    )

    system_prompt = SUPERVISOR_SYSTEM_PROMPT.format(
        customer_name=customer_data.get("name", "Unknown"),
        accounts_summary=accounts_summary or "None",
        cards_summary=cards_summary or "None",
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    messages = [{"role": "system", "content": system_prompt}]

    for turn in session.conversation_history[-10:]:
        messages.append(turn)

    messages.append({"role": "user", "content": message})

    try:
        response = await mistral_client.chat.complete_async(
            model=MISTRAL_LARGE_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        intent = IntentClassification(
            intent=data.get("intent", "clarify"),
            confidence=data.get("confidence", 0.5),
            entities=data.get("entities", {}),
            clarification_question=data.get("clarification_question"),
        )

        logger.info(
            f"[SUPERVISOR] Intent: {intent.intent.upper()} | "
            f"Confidence: {intent.confidence:.2f} | "
            f"Entities: {intent.entities} | "
            f"Routing to: {'supervisor (clarify/handoff)' if intent.intent in ('clarify', 'handoff') else intent.intent + '_agent'}"
        )

        if intent.confidence < CONFIDENCE_THRESHOLD and intent.intent not in ("clarify", "handoff"):
            logger.info(f"[SUPERVISOR] Confidence below threshold ({CONFIDENCE_THRESHOLD}), forcing clarification")
            intent.intent = "clarify"
            if not intent.clarification_question:
                intent.clarification_question = "Could you please provide more details about what you need help with?"

        return intent

    except (json.JSONDecodeError, KeyError, Exception) as e:
        logger.error(f"[SUPERVISOR] Error classifying intent: {e}")
        return IntentClassification(
            intent="clarify",
            confidence=0.0,
            entities={},
            clarification_question="I'm sorry, could you rephrase that? I want to make sure I understand your request correctly.",
        )


HANDOFF_MESSAGE = (
    "Let me connect you with a banking advisor for further assistance. "
    "A representative will be with you shortly. Your conversation history "
    "has been forwarded to ensure a smooth handoff."
)
