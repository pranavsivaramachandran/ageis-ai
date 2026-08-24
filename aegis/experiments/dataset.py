"""
Dataset validation and temporal splitting for AEGIS AI.

Ensures strict chronological ordering and prevents future data leakage
by safely partitioning historical data into train, validation, and test sets.
"""

from typing import Tuple
from datetime import timedelta
import math

from aegis.interfaces.market_data import OHLC
from aegis.experiments.models import DatasetMetadata, TemporalSplit


class DatasetBuilder:
    """
    Validates canonical historical data and produces dataset metadata.
    """
    
    @staticmethod
    def build(history: list[OHLC]) -> Tuple[DatasetMetadata, list[OHLC]]:
        """
        Validates historical data and generates metadata.
        
        Args:
            history: List of OHLC candles to validate.
            
        Returns:
            A tuple of (DatasetMetadata, validated_history).
            
        Raises:
            ValueError: If the dataset is malformed, out of order, or contains
                        invalid timestamps/symbols/timeframes.
        """
        if not history:
            raise ValueError("Dataset cannot be empty")
            
        first_candle = history[0]
        symbol = first_candle.symbol
        timeframe = first_candle.timeframe
        
        for i, candle in enumerate(history):
            # Validate symbol consistency
            if candle.symbol != symbol:
                raise ValueError(f"Mixed symbols found in dataset: {symbol} vs {candle.symbol}")
                
            # Validate timeframe consistency
            if candle.timeframe != timeframe:
                raise ValueError(f"Mixed timeframes found in dataset: {timeframe} vs {candle.timeframe}")
                
            # Validate timezone
            if candle.timestamp.tzinfo is None or candle.timestamp.utcoffset() != timedelta(0):
                raise ValueError(f"Timezone-naive or non-UTC timestamp found at index {i}")
                
            # Validate values (NaN/Inf)
            # (Pydantic Decimal handles most of this, but we explicitly reject math.nan/inf if floats sneak in)
            for val in (candle.open, candle.high, candle.low, candle.close):
                if math.isnan(float(val)) or math.isinf(float(val)):
                    raise ValueError(f"Invalid numerical value (NaN/Infinity) at index {i}")
                    
            # Validate chronological order
            if i > 0:
                prev_candle = history[i - 1]
                if candle.timestamp < prev_candle.timestamp:
                    raise ValueError(
                        f"Chronological order violated at index {i}: "
                        f"{prev_candle.timestamp} > {candle.timestamp}"
                    )
                if candle.timestamp == prev_candle.timestamp:
                    raise ValueError(
                        f"Duplicate timestamp found at index {i}: {candle.timestamp}"
                    )
                    
        metadata = DatasetMetadata(
            symbol=symbol,
            timeframe=timeframe,
            start_timestamp=history[0].timestamp,
            end_timestamp=history[-1].timestamp,
            observation_count=len(history)
        )
        
        return metadata, history


class ChronologicalSplitter:
    """
    Deterministically partitions a dataset into sequential temporal segments.
    
    Ensures TRAIN < VALIDATION < TEST to prevent future data leakage.
    Random shuffling is strictly prohibited.
    """
    
    @staticmethod
    def split(
        history: list[OHLC],
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float
    ) -> TemporalSplit:
        """
        Split a dataset chronologically into up to three contiguous segments.
        
        Args:
            history: The validated OHLC dataset.
            train_ratio: Proportion for the training set (must be > 0).
            validation_ratio: Proportion for the validation set (>= 0).
            test_ratio: Proportion for the test set (>= 0).
            
        Returns:
            A TemporalSplit containing the partitioned data.
            
        Raises:
            ValueError: If ratios are invalid or the dataset is too small
                        to satisfy the requested split.
        """
        if not history:
            raise ValueError("Cannot split an empty dataset")
            
        if train_ratio <= 0.0 or train_ratio >= 1.0:
            raise ValueError(f"train_ratio must be strictly between 0 and 1, got {train_ratio}")
            
        if validation_ratio < 0.0 or validation_ratio >= 1.0:
            raise ValueError(f"validation_ratio must be between 0 and 1, got {validation_ratio}")
            
        if test_ratio < 0.0 or test_ratio >= 1.0:
            raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")
            
        total_ratio = train_ratio + validation_ratio + test_ratio
        if not math.isclose(total_ratio, 1.0, rel_tol=1e-5):
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
            
        total_len = len(history)
        
        train_count = int(total_len * train_ratio)
        validation_count = int(total_len * validation_ratio)
        test_count = total_len - train_count - validation_count
        
        # Adjust test count if floating point rounding leaves remainder
        if test_count < 0:
            test_count = 0
            
        if train_count == 0:
            raise ValueError("Dataset is too small to yield a non-empty training set")
            
        if validation_ratio > 0.0 and validation_count == 0:
            raise ValueError("Dataset is too small to yield a non-empty validation set")
            
        if test_ratio > 0.0 and test_count == 0:
            raise ValueError("Dataset is too small to yield a non-empty test set")
            
        train_data = history[:train_count]
        validation_data = history[train_count:train_count + validation_count]
        test_data = history[train_count + validation_count:]
        
        return TemporalSplit(
            train=train_data,
            validation=validation_data,
            test=test_data
        )
