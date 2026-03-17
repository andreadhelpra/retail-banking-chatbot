from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(your|all|previous)\s+(instructions|rules|guidelines)", re.IGNORECASE),
    re.compile(r"(show|reveal|display|print|output)\s+(me\s+)?(your|the)\s+(system\s+)?prompt", re.IGNORECASE),
    re.compile(r"(pretend|act)\s+(you\s+are|as\s+if|like)", re.IGNORECASE),
    re.compile(r"(disregard|forget|override)\s+(your|all|previous)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an|in)", re.IGNORECASE),
    re.compile(r"(jailbreak|DAN|developer\s+mode)", re.IGNORECASE),
    re.compile(r"show\s+me\s+all\s+(customer|user)\s+data", re.IGNORECASE),
]

OUT_OF_SCOPE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(invest(ing|ment)?|stock(s)?|crypto(currency)?|bitcoin|ethereum|trading)\b", re.IGNORECASE),
    re.compile(r"\b(tax(es)?|tax\s+(advice|filing|return))\b", re.IGNORECASE),
    re.compile(r"\b(insurance|life\s+insurance|health\s+insurance)\b", re.IGNORECASE),
    re.compile(r"\b(mortgage|real\s+estate|property\s+(advice|investment))\b", re.IGNORECASE),
]

INPUT_REJECTION_MESSAGES = {
    "injection": "I'm sorry, but I can't process that request. I'm here to help you with your BNP Paribas banking needs. How can I assist you today?",
    "out_of_scope": "I appreciate your question, but I'm not able to provide advice on that topic. I'm specialized in retail banking services such as account inquiries, card management, and general banking information. Is there anything else I can help you with?",
}

OUTPUT_REJECTION_MESSAGE = "I apologize, but I encountered an issue generating a response. Please try rephrasing your question."


def check_input(message: str) -> dict[str, Any]:
    for pattern in INJECTION_PATTERNS:
        if pattern.search(message):
            logger.warning("[GUARDRAILS] Input check: BLOCKED | Injection detected")
            return {"allowed": False, "reason": "injection", "message": INPUT_REJECTION_MESSAGES["injection"]}
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if pattern.search(message):
            logger.warning("[GUARDRAILS] Input check: BLOCKED | Out of scope topic")
            return {"allowed": False, "reason": "out_of_scope", "message": INPUT_REJECTION_MESSAGES["out_of_scope"]}
    logger.info("[GUARDRAILS] Input check: PASS | Injection: false | Topic: in_scope")
    return {"allowed": True, "reason": None, "message": None}


def check_output(response: str, current_customer_id: str, other_customer_pii: list[str]) -> dict[str, Any]:
    for pii_item in other_customer_pii:
        if pii_item.lower() in response.lower():
            logger.warning(f"[GUARDRAILS] Output check: BLOCKED | PII leak detected: {pii_item}")
            return {"allowed": False, "reason": "pii_leak", "message": OUTPUT_REJECTION_MESSAGE}
    logger.info("[GUARDRAILS] Output check: PASS")
    return {"allowed": True, "reason": None, "message": None}


def get_other_customer_pii(banking_service, current_customer_id: str) -> list[str]:
    pii: list[str] = []
    for cid, customer in banking_service._customers.items():
        if cid == current_customer_id:
            continue
        pii.append(customer["name"])
        pii.append(cid)
        pii.append(customer.get("email", ""))
        pii.append(customer.get("phone", ""))
        for acc in customer.get("accounts", []):
            pii.append(acc["id"])
            pii.append(acc.get("iban", ""))
        for card in customer.get("cards", []):
            pii.append(card["id"])
    return [p for p in pii if p]
