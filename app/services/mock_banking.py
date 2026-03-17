from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


class MockBankingService:
    def __init__(self, data_path: str | None = None):
        if data_path is None:
            data_path = str(Path(__file__).parent.parent.parent / "data" / "mock_customers.json")
        with open(data_path) as f:
            data = json.load(f)
        self._customers: dict[str, dict[str, Any]] = {
            c["id"]: c for c in data["customers"]
        }

    def get_customer(self, customer_id: str) -> dict[str, Any] | None:
        return self._customers.get(customer_id)

    def get_customer_accounts(self, customer_id: str) -> list[dict[str, Any]]:
        customer = self.get_customer(customer_id)
        if not customer:
            return []
        return customer["accounts"]

    def get_customer_cards(self, customer_id: str) -> list[dict[str, Any]]:
        customer = self.get_customer(customer_id)
        if not customer:
            return []
        return customer["cards"]

    def get_balance(self, account_id: str) -> dict[str, Any]:
        for customer in self._customers.values():
            for account in customer["accounts"]:
                if account["id"] == account_id:
                    return {
                        "account_id": account_id,
                        "label": account["label"],
                        "type": account["type"],
                        "balance": account["balance"],
                        "currency": account["currency"],
                    }
        return {"account_id": account_id, "error": "Account not found"}

    def get_transactions(self, account_id: str, days: int = 30) -> dict[str, Any]:
        cutoff = datetime.now() - timedelta(days=days)
        for customer in self._customers.values():
            for account in customer["accounts"]:
                if account["id"] == account_id:
                    txns = [
                        t
                        for t in customer["transactions"]
                        if t["account_id"] == account_id
                        and datetime.strptime(t["date"], "%Y-%m-%d") >= cutoff
                    ]
                    txns.sort(key=lambda t: t["date"], reverse=True)
                    return {
                        "account_id": account_id,
                        "period_days": days,
                        "transactions": txns,
                    }
        return {"account_id": account_id, "error": "Account not found", "transactions": []}

    def lock_card(self, card_id: str, lock_type: str) -> dict[str, Any]:
        for customer in self._customers.values():
            for card in customer["cards"]:
                if card["id"] == card_id:
                    if card["status"] != "active":
                        return {
                            "success": False,
                            "card_id": card_id,
                            "message": f"Card is already {card['status']}. Cannot lock.",
                        }
                    new_status = "temporarily_locked" if lock_type == "temporary" else "permanently_locked"
                    card["status"] = new_status
                    return {
                        "success": True,
                        "card_id": card_id,
                        "lock_type": lock_type,
                        "message": f"Card ****{card['last_four']} has been {lock_type}ly locked.",
                    }
        return {"success": False, "card_id": card_id, "message": "Card not found"}

    def find_account_for_customer(self, customer_id: str, account_type: str | None = None) -> dict[str, Any] | None:
        accounts = self.get_customer_accounts(customer_id)
        if account_type:
            accounts = [a for a in accounts if a["type"] == account_type]
        return accounts[0] if accounts else None

    def find_card_for_customer(self, customer_id: str, card_type: str | None = None) -> list[dict[str, Any]]:
        cards = self.get_customer_cards(customer_id)
        if card_type:
            cards = [c for c in cards if c["type"] == card_type]
        return cards
