"""
Fair Value Gap (FVG) detection.

Bullish FVG: candle[i-1].high < candle[i+1].low  (gap up, price skipped this range)
Bearish FVG: candle[i-1].low > candle[i+1].high  (gap down, price skipped this range)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd


class FVGKind(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


class FVGStatus(Enum):
    OPEN = "open"
    PARTIAL = "partial"
    CLOSED = "closed"


@dataclass
class FairValueGap:
    kind: FVGKind
    formed_index: int  # index of candle[i] (middle candle)
    formed_timestamp: pd.Timestamp
    top: float    # upper bound of the gap
    bottom: float  # lower bound of the gap
    status: FVGStatus = FVGStatus.OPEN
    fill_pct: float = 0.0
    close_index: Optional[int] = None
    is_nested: bool = False  # FVG within another FVG

    @property
    def size(self) -> float:
        return self.top - self.bottom

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


def detect_fvgs(df: pd.DataFrame, min_pips: float = 2.0, pip_size: float = 0.0001) -> list[FairValueGap]:
    """
    Scan all candles and identify FVGs.
    min_pips: minimum gap size in pips (filters noise).
    pip_size: pip value (0.0001 for most FX pairs, 0.01 for JPY pairs).
    """
    highs = df["high"].values
    lows = df["low"].values
    timestamps = df.index
    min_gap = min_pips * pip_size
    fvgs: list[FairValueGap] = []

    for i in range(1, len(df) - 1):
        # Bullish FVG: gap above candle[i-1] high, below candle[i+1] low
        if lows[i + 1] > highs[i - 1]:
            gap_bottom = highs[i - 1]
            gap_top = lows[i + 1]
            if gap_top - gap_bottom >= min_gap:
                fvgs.append(
                    FairValueGap(
                        kind=FVGKind.BULLISH,
                        formed_index=i,
                        formed_timestamp=timestamps[i],
                        top=gap_top,
                        bottom=gap_bottom,
                    )
                )

        # Bearish FVG: gap below candle[i-1] low, above candle[i+1] high
        elif highs[i + 1] < lows[i - 1]:
            gap_top = lows[i - 1]
            gap_bottom = highs[i + 1]
            if gap_top - gap_bottom >= min_gap:
                fvgs.append(
                    FairValueGap(
                        kind=FVGKind.BEARISH,
                        formed_index=i,
                        formed_timestamp=timestamps[i],
                        top=gap_top,
                        bottom=gap_bottom,
                    )
                )

    return fvgs


def update_fvg_statuses(df: pd.DataFrame, fvgs: list[FairValueGap]) -> None:
    """
    Scan forward from each FVG's formation and update fill status.
    PARTIAL: price entered but did not close through the full gap.
    CLOSED: price fully filled the gap (close beyond the far edge).
    Modifies fvgs in place.
    """
    lows = df["low"].values
    highs = df["high"].values
    closes = df["close"].values

    for fvg in fvgs:
        start = fvg.formed_index + 2  # FVG is defined by i+1, so scan from i+2

        for i in range(start, len(df)):
            if fvg.status == FVGStatus.CLOSED:
                break

            low = lows[i]
            high = highs[i]
            close = closes[i]

            if fvg.kind == FVGKind.BULLISH:
                # Price retraces down into the gap
                if low <= fvg.top:
                    penetration = min(fvg.top - max(low, fvg.bottom), fvg.size)
                    fvg.fill_pct = min(penetration / fvg.size, 1.0)
                    if close <= fvg.bottom:
                        fvg.status = FVGStatus.CLOSED
                        fvg.close_index = i
                    elif fvg.fill_pct > 0:
                        fvg.status = FVGStatus.PARTIAL

            else:  # BEARISH FVG
                # Price retraces up into the gap
                if high >= fvg.bottom:
                    penetration = min(max(high, fvg.top) - fvg.bottom, fvg.size)
                    fvg.fill_pct = min(penetration / fvg.size, 1.0)
                    if close >= fvg.top:
                        fvg.status = FVGStatus.CLOSED
                        fvg.close_index = i
                    elif fvg.fill_pct > 0:
                        fvg.status = FVGStatus.PARTIAL


def mark_nested_fvgs(fvgs: list[FairValueGap]) -> None:
    """Mark FVGs that are nested inside a larger FVG of the same kind."""
    for i, fvg in enumerate(fvgs):
        for j, other in enumerate(fvgs):
            if i == j or fvg.kind != other.kind:
                continue
            if other.bottom <= fvg.bottom and fvg.top <= other.top:
                fvg.is_nested = True
                break


def get_open_fvgs(fvgs: list[FairValueGap], as_of_index: int) -> list[FairValueGap]:
    """Return FVGs that are open or partially filled as of a given candle index."""
    return [
        f for f in fvgs
        if f.formed_index < as_of_index
        and f.status in (FVGStatus.OPEN, FVGStatus.PARTIAL)
    ]
