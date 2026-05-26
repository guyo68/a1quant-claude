"""
Multi-timeframe alignment scoring.
HTF (Daily/4H) → ITF (1H) → LTF (15M/5M).
All three must align for a 3/3 setup.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd

from .structure import Trend, detect_structure, get_current_trend
from .orderblocks import OrderBlock, OBKind, detect_order_blocks, update_order_block_statuses
from .fvg import FairValueGap, FVGKind, detect_fvgs, update_fvg_statuses


RESAMPLE_MAP = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample M1 OHLCV data to a higher timeframe."""
    rule = RESAMPLE_MAP.get(timeframe, timeframe)
    resampled = df.resample(rule).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum") if "volume" in df.columns else ("close", "count"),
    ).dropna(subset=["open", "close"])
    return resampled


@dataclass
class MTFContext:
    """Snapshot of multi-timeframe alignment at a point in time."""
    htf_trend: Trend
    itf_trend: Trend
    ltf_trend: Trend
    htf_aligned: bool  # HTF has clear trend
    itf_aligned: bool  # ITF trend matches HTF
    ltf_aligned: bool  # LTF shows ChoCH confirming ITF entry
    alignment_score: float  # 0, 1/3, 2/3, or 1.0
    htf_timeframe: str
    itf_timeframe: str
    ltf_timeframe: str
    itf_ob: Optional[OrderBlock] = None  # Active OB on ITF
    itf_fvg: Optional[FairValueGap] = None  # Active FVG on ITF


def get_mtf_alignment(
    df_m1: pd.DataFrame,
    as_of: pd.Timestamp,
    htf: str = "H4",
    itf: str = "H1",
    ltf: str = "M15",
    swing_n: int = 5,
    atr_multiplier: float = 1.5,
) -> MTFContext:
    """
    Compute MTF alignment as of a specific timestamp.
    Uses only data available up to (but not including) as_of.
    """
    df_slice = df_m1[df_m1.index < as_of]
    if len(df_slice) < 100:
        return MTFContext(
            htf_trend=Trend.NEUTRAL, itf_trend=Trend.NEUTRAL, ltf_trend=Trend.NEUTRAL,
            htf_aligned=False, itf_aligned=False, ltf_aligned=False,
            alignment_score=0.0, htf_timeframe=htf, itf_timeframe=itf, ltf_timeframe=ltf,
        )

    df_htf = resample_ohlcv(df_slice, htf)
    df_itf = resample_ohlcv(df_slice, itf)
    df_ltf = resample_ohlcv(df_slice, ltf)

    htf_breaks, _, _ = detect_structure(df_htf, n=swing_n)
    itf_breaks, _, _ = detect_structure(df_itf, n=swing_n)
    ltf_breaks, _, _ = detect_structure(df_ltf, n=swing_n)

    htf_trend = get_current_trend(htf_breaks)
    itf_trend = get_current_trend(itf_breaks)
    ltf_trend = get_current_trend(ltf_breaks)

    htf_aligned = htf_trend != Trend.NEUTRAL
    itf_aligned = itf_trend == htf_trend
    ltf_aligned = ltf_trend == htf_trend

    aligned_count = sum([htf_aligned, itf_aligned, ltf_aligned])
    alignment_score = aligned_count / 3.0

    # Find active ITF order block / FVG in premium/discount zone
    itf_ob = None
    itf_fvg = None

    if len(df_itf) > 20:
        itf_obs = detect_order_blocks(df_itf, itf_breaks, atr_multiplier)
        update_order_block_statuses(df_itf, itf_obs)
        # Most recent fresh OB aligned with HTF trend
        for ob in reversed(itf_obs):
            if ob.status.value in ("fresh", "mitigated"):
                if htf_trend == Trend.BULLISH and ob.kind == OBKind.BULLISH:
                    itf_ob = ob
                    break
                elif htf_trend == Trend.BEARISH and ob.kind == OBKind.BEARISH:
                    itf_ob = ob
                    break

        itf_fvgs = detect_fvgs(df_itf)
        update_fvg_statuses(df_itf, itf_fvgs)
        for fvg in reversed(itf_fvgs):
            if fvg.status.value in ("open", "partial"):
                if htf_trend == Trend.BULLISH and fvg.kind == FVGKind.BULLISH:
                    itf_fvg = fvg
                    break
                elif htf_trend == Trend.BEARISH and fvg.kind == FVGKind.BEARISH:
                    itf_fvg = fvg
                    break

    return MTFContext(
        htf_trend=htf_trend,
        itf_trend=itf_trend,
        ltf_trend=ltf_trend,
        htf_aligned=htf_aligned,
        itf_aligned=itf_aligned,
        ltf_aligned=ltf_aligned,
        alignment_score=alignment_score,
        htf_timeframe=htf,
        itf_timeframe=itf,
        ltf_timeframe=ltf,
        itf_ob=itf_ob,
        itf_fvg=itf_fvg,
    )
