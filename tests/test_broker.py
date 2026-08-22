from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from aegis.core.config import ExecutionMode
from aegis.interfaces.broker import (
    AccountInfo,
    BrokerProvider,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)


class TestAccountInfo:

    def test_valid_account_info(self):
        account = AccountInfo(
            account_id="TEST-001",
            balance=Decimal("10000"),
            equity=Decimal("10000"),
            margin_used=Decimal("0"),
            margin_available=Decimal("10000"),
            currency="USD",
            mode=ExecutionMode.PREDICTION_ONLY,
        )

        assert account.account_id == "TEST-001"
        assert account.balance == Decimal("10000")
        assert account.mode == ExecutionMode.PREDICTION_ONLY

    def test_negative_margin_used_rejected(self):
        with pytest.raises(ValidationError):
            AccountInfo(
                account_id="TEST-001",
                balance=Decimal("10000"),
                equity=Decimal("10000"),
                margin_used=Decimal("-1"),
                margin_available=Decimal("10000"),
                mode=ExecutionMode.PREDICTION_ONLY,
            )


class TestPosition:

    def test_valid_position(self):
        position = Position(
            position_id="POS-001",
            symbol="EUR/USD",
            side=OrderSide.BUY,
            volume=Decimal("0.10"),
            entry_price=Decimal("1.10000"),
            current_price=Decimal("1.10100"),
            unrealized_pnl=Decimal("10"),
            opened_at=datetime.now(timezone.utc),
        )

        assert position.symbol == "EUR/USD"
        assert position.side == OrderSide.BUY

    def test_zero_volume_rejected(self):
        with pytest.raises(ValidationError):
            Position(
                position_id="POS-001",
                symbol="EUR/USD",
                side=OrderSide.BUY,
                volume=Decimal("0"),
                entry_price=Decimal("1.10000"),
                current_price=Decimal("1.10100"),
                unrealized_pnl=Decimal("10"),
                opened_at=datetime.now(timezone.utc),
            )

    def test_negative_price_rejected(self):
        with pytest.raises(ValidationError):
            Position(
                position_id="POS-001",
                symbol="EUR/USD",
                side=OrderSide.BUY,
                volume=Decimal("0.10"),
                entry_price=Decimal("-1"),
                current_price=Decimal("1.10100"),
                unrealized_pnl=Decimal("10"),
                opened_at=datetime.now(timezone.utc),
            )


class TestOrderRequest:

    def test_valid_market_order(self):
        order = OrderRequest(
            symbol="EUR/USD",
            side=OrderSide.BUY,
            volume=Decimal("0.10"),
            order_type=OrderType.MARKET,
        )

        assert order.order_type == OrderType.MARKET

    def test_positive_volume_required(self):
        with pytest.raises(ValidationError):
            OrderRequest(
                symbol="EUR/USD",
                side=OrderSide.BUY,
                volume=Decimal("0"),
                order_type=OrderType.MARKET,
            )

    def test_negative_volume_rejected(self):
        with pytest.raises(ValidationError):
            OrderRequest(
                symbol="EUR/USD",
                side=OrderSide.BUY,
                volume=Decimal("-1"),
                order_type=OrderType.MARKET,
            )


class TestBrokerSafety:

    def test_live_mode_is_always_rejected(self):
        class TestBroker(BrokerProvider):
            def connect(self):
                pass

            def disconnect(self):
                pass

            def get_account_info(self):
                pass

            def get_positions(self):
                return []

            def _submit_order(self, order):
                pass

            def _cancel_order(self, order_id):
                pass

            def check_health(self):
                return True

        with pytest.raises(RuntimeError, match="LIVE"):
            TestBroker(ExecutionMode.LIVE)

    def test_paper_mode_rejected_by_current_system_configuration(self):
        class TestBroker(BrokerProvider):
            def connect(self):
                pass

            def disconnect(self):
                pass

            def get_account_info(self):
                pass

            def get_positions(self):
                return []

            def _submit_order(self, order):
                pass

            def _cancel_order(self, order_id):
                pass

            def check_health(self):
                return True

        with pytest.raises(RuntimeError):
            TestBroker(ExecutionMode.PAPER)

    def test_prediction_only_broker_can_initialize(self):
        class TestBroker(BrokerProvider):
            def connect(self):
                pass

            def disconnect(self):
                pass

            def get_account_info(self):
                pass

            def get_positions(self):
                return []

            def _submit_order(self, order):
                raise AssertionError(
                    "_submit_order must never be reached in prediction-only mode"
                )

            def _cancel_order(self, order_id):
                raise AssertionError(
                    "_cancel_order must never be reached in prediction-only mode"
                )

            def check_health(self):
                return True

        broker = TestBroker(ExecutionMode.PREDICTION_ONLY)

        assert broker.mode == ExecutionMode.PREDICTION_ONLY

    def test_submit_order_blocked_in_prediction_only(self):
        class TestBroker(BrokerProvider):
            def connect(self):
                pass

            def disconnect(self):
                pass

            def get_account_info(self):
                pass

            def get_positions(self):
                return []

            def _submit_order(self, order):
                raise AssertionError(
                    "_submit_order must never be reached"
                )

            def _cancel_order(self, order_id):
                raise AssertionError(
                    "_cancel_order must never be reached"
                )

            def check_health(self):
                return True

        broker = TestBroker(ExecutionMode.PREDICTION_ONLY)

        order = OrderRequest(
            symbol="EUR/USD",
            side=OrderSide.BUY,
            volume=Decimal("0.10"),
        )

        with pytest.raises(RuntimeError, match="PREDICTION_ONLY"):
            broker.submit_order(order)

    def test_cancel_order_blocked_in_prediction_only(self):
        class TestBroker(BrokerProvider):
            def connect(self):
                pass

            def disconnect(self):
                pass

            def get_account_info(self):
                pass

            def get_positions(self):
                return []

            def _submit_order(self, order):
                raise AssertionError(
                    "_submit_order must never be reached"
                )

            def _cancel_order(self, order_id):
                raise AssertionError(
                    "_cancel_order must never be reached"
                )

            def check_health(self):
                return True

        broker = TestBroker(ExecutionMode.PREDICTION_ONLY)

        with pytest.raises(RuntimeError, match="PREDICTION_ONLY"):
            broker.cancel_order("ORDER-001")


class TestOrderResult:

    def test_valid_order_result(self):
        result = OrderResult(
            order_id="ORD-001",
            status=OrderStatus.PENDING,
            message="Order accepted",
        )

        assert result.order_id == "ORD-001"
        assert result.status == OrderStatus.PENDING

    def test_filled_order_result(self):
        result = OrderResult(
            order_id="ORD-001",
            status=OrderStatus.FILLED,
            filled_price=Decimal("1.10020"),
            filled_at=datetime.now(timezone.utc),
        )

        assert result.status == OrderStatus.FILLED
        assert result.filled_price == Decimal("1.10020")