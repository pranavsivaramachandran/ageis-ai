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
    model_config = SettingsConfigDict(env_prefix="AEGIS_", env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    # System mode
    SYSTEM_MODE: str = Field(default="PREDICTION_ONLY", description="System execution mode. Must remain PREDICTION_ONLY for Sprint 1-2.")
    
    # Analysis Configuration
    ANALYSIS_TIMEFRAME: str = Field(default="15m", description="Primary timeframe for analysis")
    
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

