"""
Backtest Engine for AEGIS AI.

Orchestrates historical replay, feature building, prediction, risk, and accounting
in a safe, look-ahead-free manner.
"""

from typing import Optional
from decimal import Decimal
from datetime import datetime

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
                    
                # 1a. Check for Stop Loss hit (intra-bar)
                stop_hit = False
                exit_price = None
                exit_reason = None
                
                if self.open_trade.stop_loss_price is not None:
                    if self.open_trade.direction == PredictionDirection.BUY and current_candle.low <= self.open_trade.stop_loss_price:
                        stop_hit = True
                        exit_price = self.open_trade.stop_loss_price
                        exit_reason = "stop_loss"
                    elif self.open_trade.direction == PredictionDirection.SELL and current_candle.high >= self.open_trade.stop_loss_price:
                        stop_hit = True
                        exit_price = self.open_trade.stop_loss_price
                        exit_reason = "stop_loss"
                        
                # 1b. Check holding period
                if not stop_hit and self.holding_ticks >= self.config.holding_period_candles and current_candle.timestamp > self.open_trade.entry_timestamp:
                    # Close trade on current candle's open
                    exit_price = current_candle.open
                    exit_reason = "holding_period_reached"
                    
                if exit_price is not None and exit_reason is not None:
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
                        exit_reason=exit_reason
                    )
                    
                    self.trades.append(completed_trade)
                    self.account = self.account.update_after_trade(pnl)
                    self.open_trade = None
                    self.holding_ticks = 0
                    
                if self.open_trade is not None:
                    # Mark-to-market tracking
                    current_price = current_candle.close
                    if self.open_trade.direction == PredictionDirection.BUY:
                        floating_pnl = (current_price - self.open_trade.entry_price) * self.open_trade.position_size
                    else:
                        floating_pnl = (self.open_trade.entry_price - current_price) * self.open_trade.position_size
                        
                    self.account = self.account.update_mtm(floating_pnl)
                    
                    floating_equity = self.account.current_equity + floating_pnl
                    if floating_equity <= Decimal("0"):
                        # Ruin
                        pnl = floating_pnl - self.config.commission_per_trade * 2
                        slippage_cost = current_price * self.config.slippage_percent * self.open_trade.position_size
                        pnl -= slippage_cost
                        
                        completed_trade = SimulatedTrade(
                            trade_id=self.open_trade.trade_id,
                            symbol=self.open_trade.symbol,
                            direction=self.open_trade.direction,
                            entry_timestamp=self.open_trade.entry_timestamp,
                            entry_price=self.open_trade.entry_price,
                            position_size=self.open_trade.position_size,
                            exit_timestamp=current_candle.timestamp,
                            exit_price=current_price,
                            realized_pnl=pnl,
                            exit_reason="liquidation_ruin"
                        )
                        self.trades.append(completed_trade)
                        self.account = self.account.update_after_trade(pnl)
                        self.open_trade = None
                        break  # End simulation

            # 2. Build Features using data strictly <= current_candle
            visible_history = history[:i+1]
            fv = self.feature_builder.build(visible_history)
            
            if fv is None:
                # Not enough data for warm-up
                continue
                
            # Check warm-up against model schema (handle upstream of ML model)
            is_warmup = False
            for req_feature in self.prediction_engine.schema.required_features:
                if getattr(fv, req_feature, None) is None:
                    is_warmup = True
                    break
                    
            if is_warmup:
                continue
                
            # 3. Predict
            prediction = self.prediction_engine.predict(fv)
            
            # 4. If we have no open trade, check risk and maybe open one
            if self.open_trade is None and prediction.direction in (PredictionDirection.BUY, PredictionDirection.SELL):
                # Phase 7: Use fv.atr_value for risk_distance. Reject if missing/zero.
                if fv.atr_value is None or fv.atr_value <= 0:
                    continue
                    
                risk_distance = Decimal(str(fv.atr_value))
                
                daily_loss, weekly_loss, monthly_loss = self._calculate_losses(current_candle.timestamp)
                
                risk_decision = self.risk_engine.evaluate_prediction(
                    prediction, 
                    risk_distance=risk_distance,
                    current_daily_loss=daily_loss,
                    current_weekly_loss=weekly_loss,
                    current_monthly_loss=monthly_loss
                )
                
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
                            
                        # Phase 5: Check margin requirement
                        required_margin = risk_decision.position_size * entry_price
                        if required_margin <= self.account.available_cash:
                            # Calculate stop loss
                            if prediction.direction == PredictionDirection.BUY:
                                stop_loss_price = entry_price - risk_distance
                            else:
                                stop_loss_price = entry_price + risk_distance
                                
                            self.open_trade = SimulatedTrade(
                                trade_id=f"t{trade_id_counter}",
                                symbol=risk_decision.symbol,
                                direction=prediction.direction,
                                entry_timestamp=next_candle.timestamp,
                                entry_price=entry_price,
                                position_size=risk_decision.position_size,
                                stop_loss_price=stop_loss_price
                            )
                            self.account = self.account.lock_margin(required_margin)
                            trade_id_counter += 1
                            self.holding_ticks = 0
                            # Note: we do NOT increment holding_ticks here, as entry happens AT next_candle.
                        
        # Force-close any open trade at the end of the backtest
        if self.open_trade is not None:
            last_candle = history[-1]
            exit_price = last_candle.close
            
            if self.open_trade.direction == PredictionDirection.BUY:
                raw_pnl = (exit_price - self.open_trade.entry_price) * self.open_trade.position_size
            else: # SELL
                raw_pnl = (self.open_trade.entry_price - exit_price) * self.open_trade.position_size
                
            pnl = raw_pnl - self.config.commission_per_trade * 2
            slippage_cost = exit_price * self.config.slippage_percent * self.open_trade.position_size
            pnl -= slippage_cost
            
            completed_trade = SimulatedTrade(
                trade_id=self.open_trade.trade_id,
                symbol=self.open_trade.symbol,
                direction=self.open_trade.direction,
                entry_timestamp=self.open_trade.entry_timestamp,
                entry_price=self.open_trade.entry_price,
                position_size=self.open_trade.position_size,
                stop_loss_price=self.open_trade.stop_loss_price,
                exit_timestamp=last_candle.timestamp,
                exit_price=exit_price,
                realized_pnl=pnl,
                exit_reason="backtest_end"
            )
            self.trades.append(completed_trade)
            self.account = self.account.update_after_trade(pnl)
            self.open_trade = None

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

    def _calculate_losses(self, current_time: datetime) -> tuple[Decimal, Decimal, Decimal]:
        """Calculates realized losses for the current day, week, and month."""
        from datetime import timedelta
        
        daily = Decimal("0")
        weekly = Decimal("0")
        monthly = Decimal("0")
        
        for t in reversed(self.trades):
            if t.exit_timestamp is None or t.realized_pnl >= 0:
                continue
                
            dt = t.exit_timestamp
            
            if dt.year == current_time.year and dt.month == current_time.month:
                monthly += abs(t.realized_pnl)
                
                if dt.isocalendar()[0:2] == current_time.isocalendar()[0:2]:
                    weekly += abs(t.realized_pnl)
                    
                if dt.day == current_time.day:
                    daily += abs(t.realized_pnl)
            elif dt < current_time - timedelta(days=31):
                break
                
        return daily, weekly, monthly
