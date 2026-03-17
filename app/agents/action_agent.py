from __future__ import annotations

import json
import logging
from typing import Any

from app.config import MISTRAL_LARGE_MODEL
from app.models.schemas import IntentClassification, PendingAction, SessionState
from app.services.mock_banking import MockBankingService

logger = logging.getLogger(__name__)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lock_card",
            "description": "Lock a bank card temporarily or permanently. Use when the customer wants to block or lock their card.",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_id": {
                        "type": "string",
                        "description": "The card ID to lock (e.g., 'card_001')",
                    },
                    "lock_type": {
                        "type": "string",
                        "enum": ["temporary", "permanent"],
                        "description": "Type of lock. Use 'temporary' for lost cards, 'permanent' for stolen cards.",
                    },
                },
                "required": ["card_id", "lock_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_balance",
            "description": "Get the current balance of a bank account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": "string",
                        "description": "The account ID to check (e.g., 'acc_001')",
                    },
                },
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_transactions",
            "description": "Get recent transactions for a bank account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": "string",
                        "description": "The account ID to check (e.g., 'acc_001')",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days of transaction history to retrieve. Default 30.",
                    },
                },
                "required": ["account_id"],
            },
        },
    },
]

ACTION_SYSTEM_PROMPT = """You are the action agent for BNP Paribas retail banking assistant.
You help customers perform banking operations using the available tools.

The user is authenticated as: {customer_name}
Their accounts: {accounts_summary}
Their cards: {cards_summary}

Rules:
- Use the tools to fulfill the customer's request.
- For balance checks, use get_balance with the appropriate account_id.
- For transaction history, use get_transactions with the account_id and number of days.
- For card locking, use lock_card with the card_id and lock_type.
- If the user says "my account" or "my balance" without specifying, use the checking account by default.
- Always use the actual account/card IDs from the customer's profile, not user-provided values.
"""


async def handle_action(
    mistral_client,
    banking_service: MockBankingService,
    message: str,
    session: SessionState,
    customer_data: dict[str, Any],
    intent: IntentClassification,
) -> dict[str, Any]:
    """Handle an action request. Returns either a confirmation request or executes the tool."""

    if session.pending_action:
        return await _handle_confirmation(message, session, banking_service)

    accounts_summary = ", ".join(
        f"{a['label']} (ID: {a['id']}, type: {a['type']})" for a in customer_data.get("accounts", [])
    )
    cards_summary = ", ".join(
        f"{c['label']} (ID: {c['id']}, ****{c['last_four']}, {c['type']}, status: {c['status']})"
        for c in customer_data.get("cards", [])
    )

    messages = [
        {
            "role": "system",
            "content": ACTION_SYSTEM_PROMPT.format(
                customer_name=customer_data.get("name", "Unknown"),
                accounts_summary=accounts_summary,
                cards_summary=cards_summary,
            ),
        },
        {"role": "user", "content": message},
    ]

    try:
        response = await mistral_client.chat.complete_async(
            model=MISTRAL_LARGE_MODEL,
            messages=messages,
            tools=TOOLS,
        )

        choice = response.choices[0]

        if choice.message.tool_calls:
            tool_call = choice.message.tool_calls[0]
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            logger.info(f"[ACTION] Tool call: {tool_name}({arguments})")

            if tool_name in ("get_balance", "get_transactions"):
                return await _execute_tool(tool_name, arguments, banking_service)

            confirmation_msg = _build_confirmation_message(tool_name, arguments, customer_data)
            session.pending_action = PendingAction(
                tool_name=tool_name,
                arguments=arguments,
                confirmation_message=confirmation_msg,
            )

            logger.info(f"[ACTION] Awaiting confirmation: {confirmation_msg}")

            return {
                "response": confirmation_msg,
                "tool_calls": [{"name": tool_name, "arguments": arguments, "status": "awaiting_confirmation"}],
            }

        return {
            "response": choice.message.content,
            "tool_calls": None,
        }

    except Exception as e:
        logger.error(f"[ACTION] Error: {e}")
        return {
            "response": "I'm experiencing a temporary issue processing your request. Please try again in a moment.",
            "tool_calls": None,
        }


async def _handle_confirmation(
    message: str,
    session: SessionState,
    banking_service: MockBankingService,
) -> dict[str, Any]:
    """Handle user confirmation of a pending action."""
    pending = session.pending_action
    affirm = message.strip().lower() in ("yes", "y", "confirm", "ok", "sure", "go ahead", "do it")
    deny = message.strip().lower() in ("no", "n", "cancel", "stop", "nevermind", "never mind")

    if affirm:
        session.pending_action = None
        return await _execute_tool(pending.tool_name, pending.arguments, banking_service)
    elif deny:
        session.pending_action = None
        logger.info(f"[ACTION] User cancelled: {pending.tool_name}")
        return {
            "response": "No problem, the operation has been cancelled. Is there anything else I can help you with?",
            "tool_calls": [{"name": pending.tool_name, "arguments": pending.arguments, "status": "cancelled"}],
        }
    else:
        return {
            "response": f"I need a clear confirmation. {pending.confirmation_message} Please reply with 'yes' or 'no'.",
            "tool_calls": [{"name": pending.tool_name, "arguments": pending.arguments, "status": "awaiting_confirmation"}],
        }


async def _execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    banking_service: MockBankingService,
) -> dict[str, Any]:
    """Execute a banking tool and format the response."""

    if tool_name == "get_balance":
        result = banking_service.get_balance(arguments["account_id"])
        if "error" in result:
            response = f"I couldn't find that account. {result['error']}"
        else:
            response = (
                f"Your {result['label']} ({result['type']} account) balance is "
                f"**{result['balance']:,.2f} {result['currency']}**."
            )

    elif tool_name == "get_transactions":
        days = arguments.get("days", 30)
        result = banking_service.get_transactions(arguments["account_id"], days)
        if "error" in result:
            response = f"I couldn't find that account. {result['error']}"
        else:
            txns = result["transactions"]
            if not txns:
                response = f"No transactions found in the last {days} days."
            else:
                lines = [f"Here are your last {len(txns)} transactions (past {days} days):\n"]
                for t in txns:
                    sign = "+" if t["amount"] > 0 else ""
                    lines.append(f"- **{t['date']}** | {t['description']} | {sign}{t['amount']:,.2f} EUR")
                response = "\n".join(lines)

    elif tool_name == "lock_card":
        result = banking_service.lock_card(arguments["card_id"], arguments["lock_type"])
        if result["success"]:
            response = f"Done! {result['message']} If you need further assistance, please don't hesitate to ask."
        else:
            response = f"I wasn't able to lock the card. {result['message']}"

    else:
        response = "I'm sorry, I don't know how to handle that operation."
        result = {}

    logger.info(f"[ACTION] Tool executed: {tool_name} | Result: success")

    return {
        "response": response,
        "tool_calls": [{"name": tool_name, "arguments": arguments, "status": "executed"}],
    }


def _build_confirmation_message(
    tool_name: str,
    arguments: dict[str, Any],
    customer_data: dict[str, Any],
) -> str:
    """Build a human-readable confirmation message for a pending action."""
    if tool_name == "lock_card":
        card_id = arguments.get("card_id", "")
        lock_type = arguments.get("lock_type", "temporary")
        card_info = ""
        for card in customer_data.get("cards", []):
            if card["id"] == card_id:
                card_info = f"{card['label']} ending in ****{card['last_four']}"
                break
        return (
            f"You'd like to **{lock_type}ly lock** your {card_info or card_id}. "
            f"Shall I proceed? (yes/no)"
        )
    return f"You'd like to execute {tool_name}. Shall I proceed? (yes/no)"
