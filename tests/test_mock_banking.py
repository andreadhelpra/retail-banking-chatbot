import pytest
from app.services.mock_banking import MockBankingService


@pytest.fixture
def bank():
    return MockBankingService()


def test_load_customers(bank):
    customer = bank.get_customer("cust_001")
    assert customer is not None
    assert customer["name"] == "Andrea Dhelpra"


def test_get_customer_not_found(bank):
    assert bank.get_customer("nonexistent") is None


def test_get_balance(bank):
    result = bank.get_balance("acc_001")
    assert result["account_id"] == "acc_001"
    assert result["balance"] == 3245.67
    assert result["currency"] == "EUR"


def test_get_balance_not_found(bank):
    result = bank.get_balance("nonexistent")
    assert result["error"] is not None


def test_get_transactions(bank):
    result = bank.get_transactions("acc_001", days=5)
    assert isinstance(result["transactions"], list)
    assert len(result["transactions"]) > 0


def test_lock_card_success(bank):
    result = bank.lock_card("card_002", "temporary")
    assert result["success"] is True
    assert result["card_id"] == "card_002"
    customer = bank.get_customer("cust_001")
    card = next(c for c in customer["cards"] if c["id"] == "card_002")
    assert card["status"] == "temporarily_locked"


def test_lock_card_not_found(bank):
    result = bank.lock_card("nonexistent", "temporary")
    assert result["success"] is False


def test_lock_card_already_locked(bank):
    bank.lock_card("card_001", "temporary")
    result = bank.lock_card("card_001", "temporary")
    assert result["success"] is False
    assert "already" in result["message"].lower()


def test_transfer_success(bank):
    result = bank.transfer("acc_001", "acc_002", 500.00)
    assert result["success"] is True
    assert result["amount"] == 500.00
    # Check balances updated
    assert result["new_from_balance"] == 3245.67 - 500.00
    assert result["new_to_balance"] == 15780.00 + 500.00


def test_transfer_insufficient_funds(bank):
    result = bank.transfer("acc_001", "acc_002", 999999.00)
    assert result["success"] is False
    assert "insufficient" in result["message"].lower()


def test_transfer_same_account(bank):
    result = bank.transfer("acc_001", "acc_001", 100.00)
    assert result["success"] is False


def test_transfer_invalid_amount(bank):
    result = bank.transfer("acc_001", "acc_002", -50.00)
    assert result["success"] is False


def test_get_customer_accounts(bank):
    accounts = bank.get_customer_accounts("cust_001")
    assert len(accounts) == 2


def test_get_customer_cards(bank):
    cards = bank.get_customer_cards("cust_001")
    assert len(cards) == 2
