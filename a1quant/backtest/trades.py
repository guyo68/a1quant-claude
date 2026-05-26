"""
Trade simulation for backtesting.
Simulates entries at OB/FVG mitigation with configurable R:R and SL placement.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd


class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"


class TradeStatus(Enum):
    OPEN = "open"
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    EXPIRED = "expired"


@dataclass
class Trade:
    trade_id: int
    direction: TradeDirection
    entry_index: int
    entry_timestamp: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_r: float = 1.0  # risk in R units (1.0 = 1R)
    setup_score: float = 0.0
    setup_notes: list = field(default_factory=list)
    status: TradeStatus = TradeStatus.OPEN
    exit_index: Optional[int] = None
    exit_timestamp: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl_r: Optional[float] = None  # realized P&L in R multiples
    max_adverse_excursion: float = 0.0  # worst drawdown during trade
    max_favorable_excursion: float = 0.0  # best profit during trade

    @property
    def risk_pips(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_pips(self) -> float:
        return abs(self.take_profit - self.entry_price)

    @property
    def rr_ratio(self) -> float:
        if self.risk_pips == 0:
            return 0.0
        return self.reward_pips / self.risk_pips


def simulate_trade(
    trade: Trade,
    df: pd.DataFrame,
    max_bars: int = 500,
) -> Trade:
    """
    Forward-simulate a trade from entry_index until TP/SL is hit or max_bars expires.
    Updates trade in place.
    """
    highs = df["high"].values
    lows = df["low"].values
    timestamps = df.index

    for i in range(trade.entry_index + 1, min(trade.entry_index + max_bars, len(df))):
        high = highs[i]
        low = lows[i]

        if trade.direction == TradeDirection.LONG:
            # Track excursions
            favorable = high - trade.entry_price
            adverse = trade.entry_price - low
            trade.max_favorable_excursion = max(trade.max_favorable_excursion, favorable)
            trade.max_adverse_excursion = max(trade.max_adverse_excursion, adverse)

            # Check stop loss first (pessimistic: assume SL hit before TP on same bar)
            if low <= trade.stop_loss:
                trade.status = TradeStatus.LOSS
                trade.exit_index = i
                trade.exit_timestamp = timestamps[i]
                trade.exit_price = trade.stop_loss
                trade.pnl_r = -trade.risk_r
                return trade
            if high >= trade.take_profit:
                trade.status = TradeStatus.WIN
                trade.exit_index = i
                trade.exit_timestamp = timestamps[i]
                trade.exit_price = trade.take_profit
                trade.pnl_r = trade.risk_r * trade.rr_ratio
                return trade

        else:  # SHORT
            favorable = trade.entry_price - low
            adverse = high - trade.entry_price
            trade.max_favorable_excursion = max(trade.max_favorable_excursion, favorable)
            trade.max_adverse_excursion = max(trade.max_adverse_excursion, adverse)

            if high >= trade.stop_loss:
                trade.status = TradeStatus.LOSS
                trade.exit_index = i
                trade.exit_timestamp = timestamps[i]
                trade.exit_price = trade.stop_loss
                trade.pnl_r = -trade.risk_r
                return trade
            if low <= trade.take_profit:
                trade.status = TradeStatus.WIN
                trade.exit_index = i
                trade.exit_timestamp = timestamps[i]
                trade.exit_price = trade.take_profit
                trade.pnl_r = trade.risk_r * trade.rr_ratio
                return trade

    # Max bars reached without resolution
    trade.status = TradeStatus.EXPIRED
    trade.exit_index = trade.entry_index + max_bars
    if trade.exit_index < len(df):
        trade.exit_timestamp = timestamps[trade.exit_index]
        trade.exit_price = df["close"].iloc[trade.exit_index]
        pnl_pips = (trade.exit_price - trade.entry_price) * (1 if trade.direction == TradeDirection.LONG else -1)
        trade.pnl_r = (pnl_pips / trade.risk_pips) if trade.risk_pips > 0 else 0.0
    return trade
