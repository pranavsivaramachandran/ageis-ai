"""
Backtest Engine for AEGIS AI.

Orchestrates historical replay, feature building, prediction, risk, and accounting
in a safe, look-ahead-free manner.
"""

from typing import Optional
from decimal import Decimal

from aegis.interfaces.market_data import OHLC
from aegis.prediction.models import PredictionDirection
from aegis.backtest.models import (
    SimulationConfig,
    VirtualAccount,
    SimulatedTrade,
    BacktestReport
)
from aegis.backtest.metrics import calculate_metrics


class BacktestEngine:
    """
    Executes a deterministic historical simulation.
    """
    
    def __init__(self, feature_builder, prediction_engine, risk_engine, config: SimulationConfig):
        self.feature_builder = feature_builder
        self.prediction_engine = prediction_engine
        self.risk_engine = risk_engine
        self.config = config
        
        self.account = VirtualAccount.initialize(config.initial_capital)
        self.trades: list[SimulatedTrade] = []
        
        self.open_trade: Optional[SimulatedTrade] = None
        self.holding_ticks = 0

    def run(self, history: list[OHLC]) -> BacktestReport:
        """Run the backtest over the provided historical candles."""
        
        self._validate_history(history)
        
        self.account = VirtualAccount.initialize(self.config.initial_capital)
        self.trades = []
        self.open_trade = None
        self.holding_ticks = 0
        
        trade_id_counter = 1
        
        for i, current_candle in enumerate(history):
            
            # 1. Process Open Trade Exits First (Exit occurs on Open if holding period reached)
            if self.open_trade is not None:
                if current_candle.timestamp > self.open_trade.entry_timestamp:
                    self.holding_ticks += 1
                    
                if self.holding_ticks >= self.config.holding_period_candles and current_candle.timestamp > self.open_trade.entry_timestamp:
                    # Close trade on current candle's open
                    exit_price = current_candle.open
                    
                    if self.open_trade.direction == PredictionDirection.BUY:
                        raw_pnl = (exit_price - self.open_trade.entry_price) * self.open_trade.position_size
                    else: # SELL
                        raw_pnl = (self.open_trade.entry_price - exit_price) * self.open_trade.position_size
                        
                    pnl = raw_pnl - self.config.commission_per_trade * 2 # entry and exit commission
                    
                    # Apply Slippage (assumes worse price)
                    slippage_cost = exit_price * self.config.slippage_percent * self.open_trade.position_size
                    pnl -= slippage_cost
                    
                    completed_trade = SimulatedTrade(
                        trade_id=self.open_trade.trade_id,
                        symbol=self.open_trade.symbol,
                        direction=self.open_trade.direction,
                        entry_timestamp=self.open_trade.entry_timestamp,
                        entry_price=self.open_trade.entry_price,
                        position_size=self.open_trade.position_size,
                        exit_timestamp=current_candle.timestamp,
                        exit_price=exit_price,
                        realized_pnl=pnl,
                        exit_reason="holding_period_reached"
                    )
                    
                    self.trades.append(completed_trade)
                    self.account = self.account.update_after_trade(pnl)
                    self.open_trade = None
                    self.holding_ticks = 0

            # 2. Build Features using data strictly <= current_candle
            visible_history = history[:i+1]
            fv = self.feature_builder.build(visible_history)
            
            if fv is None:
                # Not enough data for warm-up
                continue
                
            # 3. Predict
            prediction = self.prediction_engine.predict(fv)
            
            # 4. If we have no open trade, check risk and maybe open one
            if self.open_trade is None and prediction.direction in (PredictionDirection.BUY, PredictionDirection.SELL):
                risk_decision = self.risk_engine.evaluate_prediction(prediction, risk_distance=Decimal("1.0"))
                
                if risk_decision.status == "APPROVED":
                    # We enter on the NEXT candle's open. 
                    # If this is the last candle, we can't enter.
                    if i + 1 < len(history):
                        next_candle = history[i+1]
                        entry_price = next_candle.open
                        
                        # Apply entry slippage
                        if prediction.direction == PredictionDirection.BUY:
                            entry_price += (entry_price * self.config.slippage_percent)
                        else:
                            entry_price -= (entry_price * self.config.slippage_percent)
                            
                        self.open_trade = SimulatedTrade(
                            trade_id=f"t{trade_id_counter}",
                            symbol=risk_decision.symbol,
                            direction=prediction.direction,
                            entry_timestamp=next_candle.timestamp,
                            entry_price=entry_price,
                            position_size=risk_decision.position_size
                        )
                        trade_id_counter += 1
                        self.holding_ticks = 0
                        # Note: we do NOT increment holding_ticks here, as entry happens AT next_candle.
                        
        # 5. Finalize Report
        return calculate_metrics(
            symbol=history[0].symbol,
            timeframe=history[0].timeframe,
            start_timestamp=history[0].timestamp,
            end_timestamp=history[-1].timestamp,
            account=self.account,
            trades=self.trades
        )

    def _validate_history(self, history: list[OHLC]) -> None:
        if not history:
            raise ValueError("Empty historical data")
            
        first_candle = history[0]
        
        for i, candle in enumerate(history):
            if candle.symbol != first_candle.symbol:
                raise ValueError(f"Mixed symbols: {first_candle.symbol} and {candle.symbol}")
            if candle.timeframe != first_candle.timeframe:
                raise ValueError(f"Mixed timeframes: {first_candle.timeframe} and {candle.timeframe}")
                
            if i > 0:
                prev_candle = history[i-1]
                if candle.timestamp < prev_candle.timestamp:
                    raise ValueError(f"Candles not in chronological order: {prev_candle.timestamp} > {candle.timestamp}")
                if candle.timestamp == prev_candle.timestamp:
                    raise ValueError(f"Duplicate timestamps found: {candle.timestamp}")
