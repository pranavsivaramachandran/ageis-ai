"""
Target generation for AEGIS AI Machine Learning.

Handles leakage-free generation of targets from future OHLC data.
"""

import hashlib
from typing import Optional

from pydantic import BaseModel, Field

from aegis.interfaces.market_data import OHLC
from aegis.prediction.models import PredictionDirection


class TargetConfig(BaseModel):
    """Configuration for target generation."""
    
    target_horizon_candles: int = Field(
        default=3, 
        gt=0, 
        description="Number of candles into the future to look for the return."
    )
    threshold: float = Field(
        default=0.001, 
        ge=0.0, 
        description="Return threshold for BUY/SELL classification."
    )
    
    model_config = {"frozen": True}
    
    @property
    def identity(self) -> str:
        """Deterministic identity for this target configuration."""
        content = f"{self.target_horizon_candles}|{self.threshold}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


class TargetGenerator:
    """Generates leakage-free targets for supervised learning."""
    
    def __init__(self, config: TargetConfig):
        self.config = config
        
    def calculate_return(self, current_close: float, future_close: float) -> float:
        """Calculate simple return."""
        return (future_close - current_close) / current_close

    def generate_label(self, current_close: float, future_close: float) -> PredictionDirection:
        """
        Generate a directional label based on future return and configured threshold.
        """
        ret = self.calculate_return(current_close, future_close)
        
        # Uses strict inequality for threshold boundary
        if ret > self.config.threshold:
            return PredictionDirection.BUY
        elif ret < -self.config.threshold:
            return PredictionDirection.SELL
        else:
            return PredictionDirection.NEUTRAL
            
    def get_target(self, current_index: int, history: list[OHLC]) -> Optional[PredictionDirection]:
        """
        Get target label for a specific index in history.
        
        Args:
            current_index: The index of the candle where features are computed.
            history: The full chronological history of candles.
            
        Returns:
            The prediction direction if sufficient future data exists, else None.
        """
        target_index = current_index + self.config.target_horizon_candles
        if target_index >= len(history):
            return None
            
        current_close = float(history[current_index].close)
        future_close = float(history[target_index].close)
        
        return self.generate_label(current_close, future_close)
