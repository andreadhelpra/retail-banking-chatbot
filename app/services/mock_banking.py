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

    def transfer(self, from_account_id: str, to_account_id: str, amount: float) -> dict[str, Any]:
        """Transfer funds between accounts belonging to the same customer."""
        from_account = None
        to_account = None
        from_customer = None

        for customer in self._customers.values():
            for account in customer["accounts"]:
                if account["id"] == from_account_id:
                    from_account = account
                    from_customer = customer
                if account["id"] == to_account_id:
                    to_account = account

        if not from_account:
            return {"success": False, "message": f"Source account {from_account_id} not found."}
        if not to_account:
            return {"success": False, "message": f"Destination account {to_account_id} not found."}
        if amount <= 0:
            return {"success": False, "message": "Transfer amount must be positive."}
        if from_account["balance"] < amount:
            return {
                "success": False,
                "message": f"Insufficient funds. Available balance: {from_account['balance']:,.2f} {from_account['currency']}.",
            }
        if from_account_id == to_account_id:
            return {"success": False, "message": "Cannot transfer to the same account."}

        from_account["balance"] -= amount
        to_account["balance"] += amount

        # Record transactions
        tx_date = datetime.now().strftime("%Y-%m-%d")
        tx_id_out = f"tx_{len(from_customer['transactions']) + 1:03d}"
        from_customer["transactions"].append({
            "id": tx_id_out,
            "account_id": from_account_id,
            "date": tx_date,
            "description": f"Transfer to {to_account['label']}",
            "amount": -amount,
            "category": "transfer",
        })
        tx_id_in = f"tx_{len(from_customer['transactions']) + 1:03d}"
        from_customer["transactions"].append({
            "id": tx_id_in,
            "account_id": to_account_id,
            "date": tx_date,
            "description": f"Transfer from {from_account['label']}",
            "amount": amount,
            "category": "transfer",
        })

        return {
            "success": True,
            "from_account": from_account_id,
            "to_account": to_account_id,
            "amount": amount,
            "currency": from_account["currency"],
            "new_from_balance": from_account["balance"],
            "new_to_balance": to_account["balance"],
            "message": f"Successfully transferred {amount:,.2f} {from_account['currency']} from {from_account['label']} to {to_account['label']}.",
        }

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
