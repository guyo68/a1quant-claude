"""
Composite setup scoring (0–100).

Weights per the plan:
  MTF alignment (1/2/3 TFs)     30%
  OB/FVG quality                25%
  Liquidity sweep preceding      20%
  Premium/discount zone          15%
  Session timing                 10%

Setups ≥ 70 are A-grade signals forwarded to the LLM layer.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd

from .structure import Trend
from .orderblocks import OrderBlock, OBKind, OBStatus
from .fvg import FairValueGap, FVGKind, FVGStatus
from .liquidity import LiquidityLevel, LiquiditySide, LiquidityStatus
from .sessions import get_session_weight
from .mtf import MTFContext


AGRADE_THRESHOLD = 70


@dataclass
class SetupScore:
    total: float
    mtf_score: float
    zone_quality_score: float
    liquidity_sweep_score: float
    premium_discount_score: float
    session_score: float
    is_agrade: bool
    direction: str  # "long" or "short"
    notes: list[str]


def score_ob_quality(ob: OrderBlock) -> float:
    """
    OB quality 0–1:
    - Fresh OB: 1.0
    - Mitigated (still active): 0.6
    - Breaker: 0.4
    """
    if ob.status == OBStatus.FRESH:
        return 1.0
    elif ob.status == OBStatus.MITIGATED:
        return 0.6
    elif ob.status == OBStatus.BREAKER:
        return 0.4
    return 0.0


def score_fvg_quality(fvg: FairValueGap) -> float:
    """
    FVG quality 0–1:
    - Open: 1.0
    - Partial (< 50% filled): 0.7
    - Partial (≥ 50% filled): 0.4
    - Nested (higher conviction): bonus +0.1 capped at 1.0
    """
    if fvg.status == FVGStatus.OPEN:
        base = 1.0
    elif fvg.status == FVGStatus.PARTIAL:
        base = 0.7 if fvg.fill_pct < 0.5 else 0.4
    else:
        return 0.0
    return min(base + (0.1 if fvg.is_nested else 0.0), 1.0)


def score_premium_discount(
    price: float,
    swing_high: float,
    swing_low: float,
    direction: str,
) -> float:
    """
    Premium/discount zone alignment:
    - Long in discount (0–50%): 1.0
    - Long at equilibrium (50%): 0.5
    - Long in premium (50–100%): 0.0
    - Reversed for short.
    """
    if swing_high == swing_low:
        return 0.5
    pct = (price - swing_low) / (swing_high - swing_low)
    pct = max(0.0, min(1.0, pct))
    if direction == "long":
        return 1.0 - pct if pct <= 0.5 else 0.0
    else:
        return pct if pct >= 0.5 else 0.0


def score_liquidity_sweep(
    levels: list[LiquidityLevel],
    as_of_index: int,
    direction: str,
    lookback: int = 10,
) -> float:
    """
    Check if a liquidity sweep occurred in the last `lookback` candles
    consistent with the setup direction (sweep before reversal).
    - Long setup: sell-side liquidity (SSL) sweep just before (stops hunted below)
    - Short setup: buy-side liquidity (BSL) sweep just before
    Returns 0–1.
    """
    target_side = LiquiditySide.SELL_SIDE if direction == "long" else LiquiditySide.BUY_SIDE
    window_start = as_of_index - lookback

    for level in levels:
        if level.side != target_side:
            continue
        if level.status != LiquidityStatus.SWEPT:
            continue
        if level.sweep_index is None:
            continue
        if window_start <= level.sweep_index <= as_of_index:
            # Wick sweep (closed back inside) is higher conviction
            return 1.0 if level.close_back_inside else 0.7

    return 0.0


def score_setup(
    candle_index: int,
    candle_timestamp: pd.Timestamp,
    direction: str,
    mtf: MTFContext,
    ob: Optional[OrderBlock],
    fvg: Optional[FairValueGap],
    liquidity_levels: list[LiquidityLevel],
    current_price: float,
    swing_high: float,
    swing_low: float,
) -> SetupScore:
    """
    Compute the composite setup score (0–100).
    direction: "long" or "short"
    """
    notes: list[str] = []

    # 1. MTF alignment (30%)
    mtf_raw = mtf.alignment_score  # 0, 0.33, 0.67, or 1.0
    mtf_component = mtf_raw * 30
    notes.append(f"MTF {int(mtf.alignment_score * 3)}/3: HTF={mtf.htf_trend.value}, ITF={mtf.itf_trend.value}, LTF={mtf.ltf_trend.value}")

    # 2. OB/FVG zone quality (25%)
    zone_raw = 0.0
    if ob is not None:
        zone_raw = score_ob_quality(ob)
        notes.append(f"OB quality={zone_raw:.2f} ({ob.status.value})")
    if fvg is not None:
        fvg_q = score_fvg_quality(fvg)
        zone_raw = max(zone_raw, fvg_q)
        notes.append(f"FVG quality={fvg_q:.2f} ({fvg.status.value}, fill={fvg.fill_pct:.0%})")
    zone_component = zone_raw * 25

    # 3. Liquidity sweep (20%)
    sweep_raw = score_liquidity_sweep(liquidity_levels, candle_index, direction)
    sweep_component = sweep_raw * 20
    if sweep_raw > 0:
        notes.append(f"Liquidity sweep detected (score={sweep_raw:.2f})")
    else:
        notes.append("No recent liquidity sweep")

    # 4. Premium/discount zone (15%)
    pd_raw = score_premium_discount(current_price, swing_high, swing_low, direction)
    pd_component = pd_raw * 15
    pct_in_range = (current_price - swing_low) / max(swing_high - swing_low, 1e-10)
    notes.append(f"Price at {pct_in_range:.0%} of range ({'discount' if pct_in_range < 0.5 else 'premium'})")

    # 5. Session timing (10%)
    session_raw = get_session_weight(candle_timestamp)
    session_component = session_raw * 10
    notes.append(f"Session weight={session_raw:.2f}")

    total = mtf_component + zone_component + sweep_component + pd_component + session_component

    return SetupScore(
        total=round(total, 1),
        mtf_score=round(mtf_component, 1),
        zone_quality_score=round(zone_component, 1),
        liquidity_sweep_score=round(sweep_component, 1),
        premium_discount_score=round(pd_component, 1),
        session_score=round(session_component, 1),
        is_agrade=total >= AGRADE_THRESHOLD,
        direction=direction,
        notes=notes,
    )
