"""
Order Block detection with mitigation and breaker block tracking.

Bullish OB: last bearish candle immediately before a bullish displacement causing BOS/ChoCH.
Bearish OB: last bullish candle immediately before a bearish displacement causing BOS/ChoCH.
Displacement: move ≥ atr_multiplier × ATR(14).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd

from .structure import StructureBreak, StructureEvent


class OBKind(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


class OBStatus(Enum):
    FRESH = "fresh"
    MITIGATED = "mitigated"
    INVALIDATED = "invalidated"
    BREAKER = "breaker"  # flipped polarity


@dataclass
class OrderBlock:
    kind: OBKind
    formed_index: int
    formed_timestamp: pd.Timestamp
    high: float
    low: float
    origin_bos_index: int  # index of the BOS/ChoCH that created this OB
    status: OBStatus = OBStatus.FRESH
    mitigation_index: Optional[int] = None
    mitigation_timestamp: Optional[pd.Timestamp] = None
    invalidation_index: Optional[int] = None
    # Breaker: original OB that was mitigated and then broken through
    is_breaker: bool = False

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def detect_order_blocks(
    df: pd.DataFrame,
    structure_breaks: list[StructureBreak],
    atr_multiplier: float = 1.5,
    atr_period: int = 14,
) -> list[OrderBlock]:
    """
    For each BOS/ChoCH, find the last opposing candle before the displacement candle
    and mark it as an order block if the displacement ≥ atr_multiplier × ATR.
    """
    atr = _atr(df, atr_period)
    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    timestamps = df.index

    order_blocks: list[OrderBlock] = []

    for sb in structure_breaks:
        brk_idx = sb.index
        if brk_idx < 2:
            continue

        atr_val = atr.iloc[brk_idx]
        if pd.isna(atr_val) or atr_val == 0:
            continue

        # Upward break → look for bullish OB (last bearish candle before displacement)
        if sb.broken_level < closes[brk_idx]:
            displacement = closes[brk_idx] - opens[brk_idx]
            if displacement < atr_multiplier * atr_val:
                continue
            # Find last bearish candle before brk_idx
            ob_idx = None
            for j in range(brk_idx - 1, max(brk_idx - 20, 0) - 1, -1):
                if closes[j] < opens[j]:  # bearish candle
                    ob_idx = j
                    break
            if ob_idx is not None:
                order_blocks.append(
                    OrderBlock(
                        kind=OBKind.BULLISH,
                        formed_index=ob_idx,
                        formed_timestamp=timestamps[ob_idx],
                        high=highs[ob_idx],
                        low=lows[ob_idx],
                        origin_bos_index=brk_idx,
                    )
                )

        # Downward break → look for bearish OB (last bullish candle before displacement)
        else:
            displacement = opens[brk_idx] - closes[brk_idx]
            if displacement < atr_multiplier * atr_val:
                continue
            ob_idx = None
            for j in range(brk_idx - 1, max(brk_idx - 20, 0) - 1, -1):
                if closes[j] > opens[j]:  # bullish candle
                    ob_idx = j
                    break
            if ob_idx is not None:
                order_blocks.append(
                    OrderBlock(
                        kind=OBKind.BEARISH,
                        formed_index=ob_idx,
                        formed_timestamp=timestamps[ob_idx],
                        high=highs[ob_idx],
                        low=lows[ob_idx],
                        origin_bos_index=brk_idx,
                    )
                )

    return order_blocks


def update_order_block_statuses(
    df: pd.DataFrame, order_blocks: list[OrderBlock]
) -> None:
    """
    Scan forward from each OB's formation candle and update status:
    - MITIGATED: price retraces into zone and shows rejection (close outside zone after wick inside)
    - INVALIDATED: close fully through the OB zone
    - BREAKER: was mitigated, then price closes through it in the opposite direction
    Modifies order_blocks in place.
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    timestamps = df.index

    for ob in order_blocks:
        start = ob.origin_bos_index + 1
        mitigated = False

        for i in range(start, len(df)):
            close = closes[i]
            low = lows[i]
            high = highs[i]

            if ob.status == OBStatus.FRESH:
                # Check mitigation: wick touches zone but close is outside
                if ob.kind == OBKind.BULLISH:
                    if low <= ob.high and close > ob.high:
                        ob.status = OBStatus.MITIGATED
                        ob.mitigation_index = i
                        ob.mitigation_timestamp = timestamps[i]
                        mitigated = True
                    elif close < ob.low:
                        ob.status = OBStatus.INVALIDATED
                        ob.invalidation_index = i
                        break
                else:  # BEARISH
                    if high >= ob.low and close < ob.low:
                        ob.status = OBStatus.MITIGATED
                        ob.mitigation_index = i
                        ob.mitigation_timestamp = timestamps[i]
                        mitigated = True
                    elif close > ob.high:
                        ob.status = OBStatus.INVALIDATED
                        ob.invalidation_index = i
                        break

            elif ob.status == OBStatus.MITIGATED:
                # Check if it becomes a breaker block
                if ob.kind == OBKind.BULLISH and close < ob.low:
                    ob.status = OBStatus.BREAKER
                    ob.is_breaker = True
                    ob.invalidation_index = i
                    break
                elif ob.kind == OBKind.BEARISH and close > ob.high:
                    ob.status = OBStatus.BREAKER
                    ob.is_breaker = True
                    ob.invalidation_index = i
                    break


def get_fresh_order_blocks(
    order_blocks: list[OrderBlock], as_of_index: int
) -> list[OrderBlock]:
    """Return OBs that are fresh or mitigated (still active) as of a given candle index."""
    return [
        ob
        for ob in order_blocks
        if ob.formed_index < as_of_index
        and ob.status in (OBStatus.FRESH, OBStatus.MITIGATED)
    ]
