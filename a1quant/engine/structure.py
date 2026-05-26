"""
Swing detection, BOS (Break of Structure), and ChoCH (Change of Character).
N-candle fractal pivot: a swing high requires N candles with lower highs on each side.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd


class Trend(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class StructureEvent(Enum):
    BOS = "BOS"
    CHOCH = "ChoCH"


@dataclass
class SwingPoint:
    index: int
    timestamp: pd.Timestamp
    price: float
    kind: str  # "high" or "low"


@dataclass
class StructureBreak:
    event: StructureEvent
    timestamp: pd.Timestamp
    index: int
    price: float  # close that caused the break
    broken_level: float  # the swing level that was broken
    trend_before: Trend
    trend_after: Trend


def detect_swings(df: pd.DataFrame, n: int = 5) -> tuple[list[SwingPoint], list[SwingPoint]]:
    """
    Identify fractal swing highs and lows using an N-candle window.
    A swing high at index i: df['high'][i] is the max over [i-n, i+n].
    A swing low at index i: df['low'][i] is the min over [i-n, i+n].
    Returns (swing_highs, swing_lows).
    """
    highs: list[SwingPoint] = []
    lows: list[SwingPoint] = []
    high_arr = df["high"].values
    low_arr = df["low"].values
    timestamps = df.index

    for i in range(n, len(df) - n):
        window_h = high_arr[i - n : i + n + 1]
        window_l = low_arr[i - n : i + n + 1]

        if high_arr[i] == np.max(window_h):
            highs.append(SwingPoint(i, timestamps[i], high_arr[i], "high"))
        if low_arr[i] == np.min(window_l):
            lows.append(SwingPoint(i, timestamps[i], low_arr[i], "low"))

    return highs, lows


def detect_structure(
    df: pd.DataFrame, n: int = 5
) -> tuple[list[StructureBreak], list[SwingPoint], list[SwingPoint]]:
    """
    Detect BOS and ChoCH events from swing points.

    Rules:
    - Trend starts NEUTRAL.
    - A close above the most recent swing high while bullish → BOS (continuation).
    - A close above the most recent swing high while bearish → ChoCH (reversal, trend flips bullish).
    - A close below the most recent swing low while bearish → BOS (continuation).
    - A close below the most recent swing low while bullish → ChoCH (reversal, trend flips bearish).

    Returns (structure_breaks, swing_highs, swing_lows).
    """
    swing_highs, swing_lows = detect_swings(df, n)

    closes = df["close"].values
    timestamps = df.index
    events: list[StructureBreak] = []
    trend = Trend.NEUTRAL

    # Build sorted combined list of swing points for sequential scanning
    all_swings = sorted(swing_highs + swing_lows, key=lambda s: s.index)

    # Track last confirmed swing high/low for BOS/ChoCH checks
    last_high: Optional[SwingPoint] = None
    last_low: Optional[SwingPoint] = None

    # Swing point lookup by index for fast access
    high_by_idx = {s.index: s for s in swing_highs}
    low_by_idx = {s.index: s for s in swing_lows}

    for i in range(1, len(closes)):
        close = closes[i]

        # Update last confirmed swing points up to candle i (confirmed = index ≤ i-n)
        if i in high_by_idx:
            last_high = high_by_idx[i]
        if i in low_by_idx:
            last_low = low_by_idx[i]

        if last_high is None or last_low is None:
            continue

        # Check for upward break
        if close > last_high.price:
            if trend == Trend.BEARISH:
                event_type = StructureEvent.CHOCH
                new_trend = Trend.BULLISH
            elif trend == Trend.BULLISH:
                event_type = StructureEvent.BOS
                new_trend = Trend.BULLISH
            else:
                event_type = StructureEvent.BOS
                new_trend = Trend.BULLISH

            events.append(
                StructureBreak(
                    event=event_type,
                    timestamp=timestamps[i],
                    index=i,
                    price=close,
                    broken_level=last_high.price,
                    trend_before=trend,
                    trend_after=new_trend,
                )
            )
            trend = new_trend
            # Reset last_high so next break needs a new swing
            last_high = None

        # Check for downward break
        elif close < last_low.price:
            if trend == Trend.BULLISH:
                event_type = StructureEvent.CHOCH
                new_trend = Trend.BEARISH
            elif trend == Trend.BEARISH:
                event_type = StructureEvent.BOS
                new_trend = Trend.BEARISH
            else:
                event_type = StructureEvent.BOS
                new_trend = Trend.BEARISH

            events.append(
                StructureBreak(
                    event=event_type,
                    timestamp=timestamps[i],
                    index=i,
                    price=close,
                    broken_level=last_low.price,
                    trend_before=trend,
                    trend_after=new_trend,
                )
            )
            trend = new_trend
            last_low = None

    return events, swing_highs, swing_lows


def get_current_trend(structure_breaks: list[StructureBreak]) -> Trend:
    """Return the most recent trend from structure break history."""
    if not structure_breaks:
        return Trend.NEUTRAL
    return structure_breaks[-1].trend_after
