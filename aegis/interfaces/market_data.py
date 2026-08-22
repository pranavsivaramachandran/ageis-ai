"""
Market data abstraction for AEGIS AI.

Defines provider-independent canonical data models:
- Tick
- OHLC
- Timeframe

and the abstract MarketDataProvider interface.

Prices use Decimal for financial precision.
All timestamps must be timezone-aware UTC datetimes.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional, Self

from pydantic import BaseModel, Field, model_validator


class Timeframe(str, Enum):
    """Standardized candle timeframe periods."""

    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"


class Tick(BaseModel):
    """
    A single price snapshot for an instrument.

    Canonical internal representation — all providers normalize to this.
    """

    symbol: str = Field(
        ...,
        min_length=1,
        description="Instrument symbol e.g. 'EUR/USD'",
    )

    timestamp: datetime = Field(
        ...,
        description="UTC timestamp of the tick",
    )

    bid: Decimal = Field(
        ...,
        gt=0,
        description="Best bid price",
    )

    ask: Decimal = Field(
        ...,
        gt=0,
        description="Best ask price",
    )

    volume: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Tick volume if available",
    )

    model_config = {"frozen": True}

    @property
    def mid(self) -> Decimal:
        """Mid-price calculated as average of bid and ask."""

        return (self.bid + self.ask) / Decimal("2")

    @model_validator(mode="after")
    def validate_tick(self) -> Self:
        """Validate tick timestamp and bid/ask relationship."""

        if self.timestamp.tzinfo is None:
            raise ValueError(
                "Tick timestamp must be timezone-aware UTC"
            )

        if self.timestamp.utcoffset() != timedelta(0):
            raise ValueError(
                "Tick timestamp must be in UTC"
            )

        if self.bid > self.ask:
            raise ValueError(
                f"Bid ({self.bid}) cannot exceed ask ({self.ask})"
            )

        return self

    def is_stale(self, max_age_seconds: int = 60) -> bool:
        """Return True if the tick is older than max_age_seconds."""

        if max_age_seconds < 0:
            raise ValueError(
                "max_age_seconds cannot be negative"
            )

        age = datetime.now(timezone.utc) - self.timestamp

        return age > timedelta(seconds=max_age_seconds)


class OHLC(BaseModel):
    """
    A single OHLC candlestick bar.

    Canonical internal representation for candle data.
    """

    symbol: str = Field(
        ...,
        min_length=1,
        description="Instrument symbol",
    )

    timestamp: datetime = Field(
        ...,
        description="UTC timestamp of the candle open",
    )

    timeframe: Timeframe = Field(
        ...,
        description="Candle timeframe period",
    )

    open: Decimal = Field(
        ...,
        gt=0,
        description="Opening price",
    )

    high: Decimal = Field(
        ...,
        gt=0,
        description="Highest price in period",
    )

    low: Decimal = Field(
        ...,
        gt=0,
        description="Lowest price in period",
    )

    close: Decimal = Field(
        ...,
        gt=0,
        description="Closing price",
    )

    volume: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Volume if available",
    )

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def validate_ohlc(self) -> Self:
        """Validate timestamp and OHLC price relationships."""

        if self.timestamp.tzinfo is None:
            raise ValueError(
                "OHLC timestamp must be timezone-aware UTC"
            )

        if self.timestamp.utcoffset() != timedelta(0):
            raise ValueError(
                "OHLC timestamp must be in UTC"
            )

        if self.high < self.open:
            raise ValueError(
                f"High ({self.high}) cannot be below open ({self.open})"
            )

        if self.high < self.close:
            raise ValueError(
                f"High ({self.high}) cannot be below close ({self.close})"
            )

        if self.low > self.open:
            raise ValueError(
                f"Low ({self.low}) cannot be above open ({self.open})"
            )

        if self.low > self.close:
            raise ValueError(
                f"Low ({self.low}) cannot be above close ({self.close})"
            )

        if self.high < self.low:
            raise ValueError(
                f"High ({self.high}) cannot be below low ({self.low})"
            )

        return self


class MarketDataProvider(ABC):
    """
    Abstract interface for market data providers.

    Concrete implementations will be added in future sprints.
    All providers must normalize their data into the canonical
    Tick and OHLC models.
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the data provider."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the data provider."""

    @abstractmethod
    def get_latest_tick(self, symbol: str) -> Tick:
        """Retrieve the most recent tick for a symbol."""

    @abstractmethod
    def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: int,
    ) -> list[OHLC]:
        """Retrieve the last count candles."""

    @abstractmethod
    def check_health(self) -> bool:
        """Return True if the provider connection is healthy."""