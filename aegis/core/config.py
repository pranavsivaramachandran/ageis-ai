from decimal import Decimal
from enum import Enum
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator
from typing import Self


class ExecutionMode(str, Enum):
    """
    Defines the three execution tiers for AEGIS AI.
    PREDICTION_ONLY: No orders of any kind. Analysis and predictions only.
    PAPER: Simulated orders against a paper account (future sprint).
    LIVE: Real-money execution against a live broker (future sprint, heavily gated).
    """
    PREDICTION_ONLY = "PREDICTION_ONLY"
    PAPER = "PAPER"
    LIVE = "LIVE"


class AegisConfig(BaseSettings):
    """
    Core configuration for the AEGIS AI platform.
    Uses environment variables for secure, flexible configuration.
    """
    model_config = SettingsConfigDict(env_prefix="AEGIS_", env_file=".env", env_file_encoding="utf-8", extra="ignore", frozen=True)
    
    # System mode
    SYSTEM_MODE: str = Field(default="PREDICTION_ONLY", description="System execution mode. Must remain PREDICTION_ONLY for Sprint 1-2.")
    
    # Analysis Configuration
    ANALYSIS_TIMEFRAME: str = Field(default="15m", description="Primary timeframe for analysis")
    
    # Risk Management Configuration
    RISK_MIN_CONFIDENCE: Decimal = Field(default=Decimal("0.7"), description="Minimum confidence required for risk approval.")
    RISK_MAX_DAILY_LOSS: Decimal = Field(default=Decimal("1000.0"), description="Maximum daily realized loss allowed.")
    RISK_MAX_WEEKLY_LOSS: Decimal = Field(default=Decimal("5000.0"), description="Maximum weekly realized loss allowed.")
    RISK_MAX_MONTHLY_LOSS: Decimal = Field(default=Decimal("15000.0"), description="Maximum monthly realized loss allowed.")
    RISK_MAX_RISK_PER_TRADE: Decimal = Field(default=Decimal("100.0"), description="Maximum risk amount per trade.")
    RISK_MAX_POSITION_SIZE: Decimal = Field(default=Decimal("10.0"), description="Maximum position size allowed.")

    # Database
    DATABASE_URL: str = Field(default="sqlite:///aegis.db", description="Database connection string")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Global logging level")

    @model_validator(mode='after')
    def check_system_mode(self) -> Self:
        if self.SYSTEM_MODE != ExecutionMode.PREDICTION_ONLY.value:
            raise ValueError("CRITICAL: System mode must be PREDICTION_ONLY. Live trading is strictly prohibited in this phase.")
        return self

# Global instance of configuration
config = AegisConfig()

