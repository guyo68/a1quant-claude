"""
Liquidity mapping: equal highs/lows, PDH/PDL, PWH/PWL, session ranges, liquidity sweeps.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np

from .structure import SwingPoint


class LiquiditySide(Enum):
    BUY_SIDE = "buy_side"   # above highs — retail long stop losses
    SELL_SIDE = "sell_side"  # below lows — retail short stop losses


class LiquidityStatus(Enum):
    INTACT = "intact"
    SWEPT = "swept"


@dataclass
class LiquidityLevel:
    side: LiquiditySide
    price: float
    formed_index: int
    formed_timestamp: pd.Timestamp
    label: str  # e.g. "equal_high", "PDH", "PWH", "session_high"
    touch_count: int = 1
    status: LiquidityStatus = LiquidityStatus.INTACT
    sweep_index: Optional[int] = None
    sweep_timestamp: Optional[pd.Timestamp] = None
    close_back_inside: bool = False  # wick sweep: price closed back inside after sweep


@dataclass
class SessionRange:
    name: str  # "asia", "london", "new_york"
    date: pd.Timestamp
    high: float
    low: float
    high_index: int
    low_index: int
    high_swept: bool = False
    low_swept: bool = False


def detect_equal_levels(
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    tolerance_pips: float = 3.0,
    pip_size: float = 0.0001,
) -> list[LiquidityLevel]:
    """
    Identify equal highs and equal lows (within tolerance_pips of each other).
    Multiple equal highs cluster into a single buy-side liquidity level.
    """
    tolerance = tolerance_pips * pip_size
    levels: list[LiquidityLevel] = []

    def cluster_points(points: list[SwingPoint], side: LiquiditySide, label: str) -> None:
        prices = np.array([p.price for p in points])
        used = [False] * len(points)
        for i in range(len(points)):
            if used[i]:
                continue
            cluster = [points[i]]
            used[i] = True
            for j in range(i + 1, len(points)):
                if not used[j] and abs(prices[i] - prices[j]) <= tolerance:
                    cluster.append(points[j])
                    used[j] = True
            if len(cluster) >= 2:
                avg_price = np.mean([p.price for p in cluster])
                first = min(cluster, key=lambda p: p.index)
                levels.append(
                    LiquidityLevel(
                        side=side,
                        price=avg_price,
                        formed_index=first.index,
                        formed_timestamp=first.timestamp,
                        label=label,
                        touch_count=len(cluster),
                    )
                )

    cluster_points(swing_highs, LiquiditySide.BUY_SIDE, "equal_high")
    cluster_points(swing_lows, LiquiditySide.SELL_SIDE, "equal_low")
    return levels


def detect_pdh_pdl(df: pd.DataFrame) -> list[LiquidityLevel]:
    """
    Previous Day High/Low as liquidity levels.
    Requires df indexed by timestamp with at least 2 days of data.
    """
    levels: list[LiquidityLevel] = []
    if not isinstance(df.index, pd.DatetimeIndex):
        return levels

    daily = df.resample("1D").agg({"high": "max", "low": "min", "close": "last"})
    daily = daily.dropna()

    for i in range(1, len(daily)):
        day = daily.index[i]
        prev_day = daily.index[i - 1]
        pdh = daily["high"].iloc[i - 1]
        pdl = daily["low"].iloc[i - 1]

        # Find the index in df for start of current day
        day_mask = df.index.date == day.date()
        if not day_mask.any():
            continue
        formed_idx = df.index.get_loc(df[day_mask].index[0])

        levels.append(
            LiquidityLevel(
                side=LiquiditySide.BUY_SIDE,
                price=pdh,
                formed_index=formed_idx,
                formed_timestamp=day,
                label="PDH",
            )
        )
        levels.append(
            LiquidityLevel(
                side=LiquiditySide.SELL_SIDE,
                price=pdl,
                formed_index=formed_idx,
                formed_timestamp=day,
                label="PDL",
            )
        )

    return levels


def detect_session_ranges(df: pd.DataFrame, tz: str = "America/New_York") -> list[SessionRange]:
    """
    Compute Asia / London / NY session high+low for each day.
    Session times in EST/EDT:
      Asia:    20:00–00:00 (previous day 20:00 to current day 00:00)
      London:  02:00–05:00
      NY:      07:00–10:00
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return []

    df_tz = df.copy()
    if df_tz.index.tz is None:
        df_tz.index = df_tz.index.tz_localize("UTC")
    df_tz.index = df_tz.index.tz_convert(tz)

    sessions: list[SessionRange] = []
    dates = pd.Series(df_tz.index.date).unique()

    session_defs = {
        "london": (2, 5),
        "new_york": (7, 10),
    }

    for date in dates:
        date_ts = pd.Timestamp(date)
        for session_name, (start_h, end_h) in session_defs.items():
            mask = (
                (df_tz.index.date == date)
                & (df_tz.index.hour >= start_h)
                & (df_tz.index.hour < end_h)
            )
            session_df = df_tz[mask]
            if session_df.empty or len(session_df) < 3:
                continue
            high_iloc = session_df["high"].idxmax()
            low_iloc = session_df["low"].idxmin()
            sessions.append(
                SessionRange(
                    name=session_name,
                    date=date_ts,
                    high=session_df["high"].max(),
                    low=session_df["low"].min(),
                    high_index=df.index.get_loc(high_iloc.tz_localize(None) if high_iloc.tzinfo else high_iloc),
                    low_index=df.index.get_loc(low_iloc.tz_localize(None) if low_iloc.tzinfo else low_iloc),
                )
            )

    return sessions


def detect_liquidity_sweeps(
    df: pd.DataFrame,
    levels: list[LiquidityLevel],
    tolerance_pips: float = 1.0,
    pip_size: float = 0.0001,
) -> None:
    """
    Scan forward from each level's formation and detect liquidity sweeps.
    A sweep = price wicks beyond the level then closes back inside.
    Modifies levels in place.
    """
    tolerance = tolerance_pips * pip_size
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    timestamps = df.index

    for level in levels:
        start = level.formed_index + 1

        for i in range(start, len(df)):
            if level.status == LiquidityStatus.SWEPT:
                break

            if level.side == LiquiditySide.BUY_SIDE:
                if highs[i] > level.price + tolerance:
                    level.status = LiquidityStatus.SWEPT
                    level.sweep_index = i
                    level.sweep_timestamp = timestamps[i]
                    # Wick sweep: closed back below the level
                    level.close_back_inside = closes[i] < level.price
            else:  # SELL_SIDE
                if lows[i] < level.price - tolerance:
                    level.status = LiquidityStatus.SWEPT
                    level.sweep_index = i
                    level.sweep_timestamp = timestamps[i]
                    level.close_back_inside = closes[i] > level.price
