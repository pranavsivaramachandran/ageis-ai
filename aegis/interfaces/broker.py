"""
Broker abstraction for AEGIS AI.

Defines provider-independent broker models (AccountInfo, Position, Order*)
and the abstract BrokerProvider interface with built-in safety gates.

Safety architecture (two independent layers):
  1. config.model_validator rejects non-PREDICTION_ONLY at startup
  2. BrokerProvider.__init__() validates mode against config
  3. BrokerProvider.submit_order() pre-check blocks in PREDICTION_ONLY

LIVE mode is always rejected regardless of configuration.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from aegis.core.config import ExecutionMode, config
from aegis.core.logger import get_logger

logger = get_logger(__name__)


class OrderSide(str, Enum):
    """Trade direction."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order execution mechanism."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(str, Enum):
    """Order lifecycle states."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class AccountInfo(BaseModel):
    """Snapshot of a broker account."""
    account_id: str = Field(..., min_length=1, description="Broker account identifier")
    balance: Decimal = Field(..., description="Account balance")
    equity: Decimal = Field(..., description="Account equity (balance + unrealized PnL)")
    margin_used: Decimal = Field(default=Decimal("0"), ge=0, description="Margin currently in use")
    margin_available: Decimal = Field(default=Decimal("0"), ge=0, description="Available margin")
    currency: str = Field(default="USD", min_length=1, description="Account currency")
    mode: ExecutionMode = Field(..., description="Execution mode this account is operating under")

    model_config = {"frozen": True}


class Position(BaseModel):
    """An open trading position."""
    position_id: str = Field(..., min_length=1, description="Unique position identifier")
    symbol: str = Field(..., min_length=1, description="Instrument symbol")
    side: OrderSide = Field(..., description="Position direction")
    volume: Decimal = Field(..., gt=0, description="Position size")
    entry_price: Decimal = Field(..., gt=0, description="Entry price")
    current_price: Decimal = Field(..., gt=0, description="Current market price")
    unrealized_pnl: Decimal = Field(..., description="Unrealized profit/loss")
    opened_at: datetime = Field(..., description="UTC timestamp when position was opened")

    model_config = {"frozen": True}


class OrderRequest(BaseModel):
    """Intent to submit an order to a broker."""
    symbol: str = Field(..., min_length=1, description="Instrument symbol")
    side: OrderSide = Field(..., description="Order direction")
    volume: Decimal = Field(..., gt=0, description="Order size in lots")
    order_type: OrderType = Field(default=OrderType.MARKET, description="Order execution type")
    price: Optional[Decimal] = Field(default=None, gt=0, description="Limit/Stop price (required for non-MARKET orders)")
    stop_loss: Optional[Decimal] = Field(default=None, gt=0, description="Stop loss price")
    take_profit: Optional[Decimal] = Field(default=None, gt=0, description="Take profit price")

    model_config = {"frozen": True}


class OrderResult(BaseModel):
    """Outcome of an order submission or cancellation."""
    order_id: str = Field(..., min_length=1, description="Broker-assigned order identifier")
    status: OrderStatus = Field(..., description="Current order status")
    filled_price: Optional[Decimal] = Field(default=None, description="Price at which order was filled")
    filled_at: Optional[datetime] = Field(default=None, description="UTC timestamp of fill")
    message: str = Field(default="", description="Broker message or error detail")

    model_config = {"frozen": True}


class BrokerProvider(ABC):
    """
    Abstract interface for broker providers.

    Safety gates:
      - __init__() validates that the requested mode is compatible with config.SYSTEM_MODE
      - LIVE mode is ALWAYS rejected regardless of configuration
      - submit_order() raises RuntimeError in PREDICTION_ONLY mode before reaching the provider

    Concrete implementations (e.g., PaperBroker, MT5Broker) will subclass this
    in future sprints.
    """

    def __init__(self, mode: ExecutionMode):
        # Gate 1: LIVE mode is always rejected
        if mode == ExecutionMode.LIVE:
            raise RuntimeError(
                "CRITICAL: LIVE execution mode is not available. "
                "Live trading requires explicit future-sprint authorization."
            )

        # Gate 2: Requested mode must match system config
        config_mode = ExecutionMode(config.SYSTEM_MODE)
        if mode != config_mode:
            raise RuntimeError(
                f"CRITICAL: Broker mode '{mode.value}' conflicts with system mode '{config_mode.value}'. "
                f"Cannot initialize broker in a mode above the system execution tier."
            )

        self._mode = mode
        logger.info("BrokerProvider initialized", mode=mode.value)

    @property
    def mode(self) -> ExecutionMode:
        """The execution mode this broker is operating under."""
        return self._mode

    def submit_order(self, order: OrderRequest) -> OrderResult:
        """
        Submit an order. Includes a safety pre-check before delegating
        to the concrete implementation.

        In PREDICTION_ONLY mode, this ALWAYS raises before reaching _submit_order().
        """
        if self._mode == ExecutionMode.PREDICTION_ONLY:
            raise RuntimeError(
                "CRITICAL: Order submission is blocked in PREDICTION_ONLY mode. "
                "No orders (real or simulated) are permitted."
            )
        return self._submit_order(order)

    def cancel_order(self, order_id: str) -> OrderResult:
        """
        Cancel an order. Includes a safety pre-check before delegating
        to the concrete implementation.

        In PREDICTION_ONLY mode, this ALWAYS raises before reaching _cancel_order().
        """
        if self._mode == ExecutionMode.PREDICTION_ONLY:
            raise RuntimeError(
                "CRITICAL: Order cancellation is blocked in PREDICTION_ONLY mode."
            )
        return self._cancel_order(order_id)

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the broker."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the broker."""
        pass

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """Retrieve current account information."""
        pass

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Retrieve all open positions."""
        pass

    @abstractmethod
    def _submit_order(self, order: OrderRequest) -> OrderResult:
        """Internal order submission — called only after safety pre-checks pass."""
        pass

    @abstractmethod
    def _cancel_order(self, order_id: str) -> OrderResult:
        """Internal order cancellation — called only after safety pre-checks pass."""
        pass

    @abstractmethod
    def check_health(self) -> bool:
        """Return True if the broker connection is healthy."""
        pass
