"""
Backtest runner: orchestrates the full pipeline on historical data.

Strategy:
  1. Pre-compute all structural analysis on the full dataset (O(n)).
  2. Walk forward bar-by-bar, filtering structures by formed_index < current_bar
     so no future data leaks into the decision at any point.
  3. Simulate trades forward from entry bar using simulate_trade().
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

from ..engine.structure import detect_structure, get_current_trend, Trend
from ..engine.orderblocks import (
    detect_order_blocks, update_order_block_statuses, get_fresh_order_blocks,
    OBKind, OBStatus
)
from ..engine.fvg import detect_fvgs, update_fvg_statuses, get_open_fvgs, FVGKind
from ..engine.liquidity import detect_equal_levels, detect_liquidity_sweeps, LiquidityLevel
from ..engine.mtf import resample_ohlcv
from ..engine.scoring import score_setup, AGRADE_THRESHOLD
from ..engine.structure import detect_structure
from .trades import Trade, TradeDirection, TradeStatus, simulate_trade
from .metrics import compute_metrics, BacktestMetrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    rr_ratio: float = 2.0
    min_score: float = 60.0
    max_concurrent_trades: int = 1
    swing_n: int = 5
    atr_multiplier: float = 1.5
    atr_period: int = 14
    fvg_min_pips: float = 2.0
    pip_size: float = 0.0001
    max_bars_in_trade: int = 500
    htf: str = "H4"
    itf: str = "H1"
    ltf: str = "M15"
    sweep_lookback: int = 20
    session_filter: Optional[list[str]] = None


def _build_htf_trend_series(df: pd.DataFrame, timeframe: str, swing_n: int) -> pd.Series:
    """
    Pre-compute HTF trend at each M1 bar by detecting structure on the resampled
    frame and forward-filling the trend value down to M1 resolution.
    Returns a Series indexed like df with Trend enum values.
    """
    df_htf = resample_ohlcv(df, timeframe)
    breaks, _, _ = detect_structure(df_htf, n=swing_n)

    # Build a trend series at HTF resolution
    htf_trends = pd.Series(Trend.NEUTRAL, index=df_htf.index)
    current = Trend.NEUTRAL
    for b in breaks:
        current = b.trend_after
        htf_trends.iloc[b.index] = current

    # Forward-fill to every HTF bar, then reindex to M1
    htf_trends = htf_trends.ffill()
    # Reindex to M1 frequency and ffill gaps
    m1_trends = htf_trends.reindex(df.index, method="ffill")
    m1_trends = m1_trends.ffill().fillna(Trend.NEUTRAL)
    return m1_trends


def _build_itf_trend_series(df: pd.DataFrame, timeframe: str, swing_n: int) -> pd.Series:
    """Same as _build_htf_trend_series but for ITF."""
    return _build_htf_trend_series(df, timeframe, swing_n)


def run_backtest(
    df: pd.DataFrame,
    config: BacktestConfig = None,
    warmup_bars: int = 500,
    verbose: bool = False,
) -> tuple[list[Trade], BacktestMetrics]:
    """
    Run a full bar-by-bar backtest on M1 OHLCV data.
    Pre-computes all structural data once (O(n)), then walks forward.
    Returns (trades, metrics).
    """
    if config is None:
        config = BacktestConfig()

    logger.info(f"Starting backtest: {len(df):,} bars, warmup={warmup_bars}")

    # ── Pre-compute everything once ──────────────────────────────────────────
    logger.info("Pre-computing structure...")
    sb_full, sw_highs, sw_lows = detect_structure(df, n=config.swing_n)

    logger.info("Pre-computing order blocks...")
    obs_full = detect_order_blocks(df, sb_full, config.atr_multiplier, config.atr_period)
    update_order_block_statuses(df, obs_full)

    logger.info("Pre-computing FVGs...")
    fvgs_full = detect_fvgs(df, config.fvg_min_pips, config.pip_size)
    update_fvg_statuses(df, fvgs_full)

    logger.info("Pre-computing liquidity levels...")
    eq_levels = detect_equal_levels(sw_highs, sw_lows, pip_size=config.pip_size)
    detect_liquidity_sweeps(df, eq_levels, pip_size=config.pip_size)

    logger.info("Pre-computing HTF/ITF trend series...")
    htf_trends = _build_htf_trend_series(df, config.htf, config.swing_n)
    itf_trends = _build_itf_trend_series(df, config.itf, config.swing_n)
    ltf_trends = _build_itf_trend_series(df, config.ltf, config.swing_n)

    # HTF swing range (rolling 20-bar H4 high/low for premium/discount scoring)
    df_htf = resample_ohlcv(df, config.htf)
    htf_roll_high = df_htf["high"].rolling(20, min_periods=1).max().reindex(df.index, method="ffill").ffill()
    htf_roll_low = df_htf["low"].rolling(20, min_periods=1).min().reindex(df.index, method="ffill").ffill()

    # ── Walk-forward simulation ───────────────────────────────────────────────
    closes = df["close"].values
    timestamps = df.index
    trades: list[Trade] = []
    trade_counter = 0
    open_trades: list[Trade] = []

    logger.info("Walking forward...")

    for i in range(warmup_bars, len(df) - 1):
        # Resolve open trades that may have hit SL/TP
        still_open = []
        for trade in open_trades:
            updated = simulate_trade(trade, df, max_bars=config.max_bars_in_trade)
            if updated.status != TradeStatus.OPEN:
                trades.append(updated)
                if verbose:
                    logger.info(f"  T{updated.trade_id} {updated.status.value} {updated.pnl_r:+.2f}R")
            else:
                still_open.append(updated)
        open_trades = still_open

        if len(open_trades) >= config.max_concurrent_trades:
            continue

        # HTF trend at this bar (pre-computed, no look-ahead)
        htf_trend = htf_trends.iloc[i]
        if htf_trend == Trend.NEUTRAL:
            continue

        direction = "long" if htf_trend == Trend.BULLISH else "short"
        current_price = closes[i]

        # Find an OB or FVG that formed before this bar and price is touching
        active_obs = get_fresh_order_blocks(obs_full, i)
        active_fvgs = get_open_fvgs(fvgs_full, i)

        target_ob = None
        target_fvg = None
        entry_zone_high = None
        entry_zone_low = None

        for ob in reversed(active_obs):
            if direction == "long" and ob.kind == OBKind.BULLISH and ob.contains(current_price):
                target_ob = ob
                entry_zone_high, entry_zone_low = ob.high, ob.low
                break
            elif direction == "short" and ob.kind == OBKind.BEARISH and ob.contains(current_price):
                target_ob = ob
                entry_zone_high, entry_zone_low = ob.high, ob.low
                break

        if target_ob is None:
            for fvg in reversed(active_fvgs):
                if direction == "long" and fvg.kind == FVGKind.BULLISH and fvg.contains(current_price):
                    target_fvg = fvg
                    entry_zone_high, entry_zone_low = fvg.top, fvg.bottom
                    break
                elif direction == "short" and fvg.kind == FVGKind.BEARISH and fvg.contains(current_price):
                    target_fvg = fvg
                    entry_zone_high, entry_zone_low = fvg.top, fvg.bottom
                    break

        if target_ob is None and target_fvg is None:
            continue

        swing_high = float(htf_roll_high.iloc[i])
        swing_low = float(htf_roll_low.iloc[i])
        if math.isnan(swing_high) or math.isnan(swing_low):
            continue

        # Build a lightweight MTFContext from pre-computed trend series
        from ..engine.mtf import MTFContext
        mtf = MTFContext(
            htf_trend=htf_trend,
            itf_trend=itf_trends.iloc[i],
            ltf_trend=ltf_trends.iloc[i],
            htf_aligned=htf_trend != Trend.NEUTRAL,
            itf_aligned=itf_trends.iloc[i] == htf_trend,
            ltf_aligned=ltf_trends.iloc[i] == htf_trend,
            alignment_score=sum([
                htf_trend != Trend.NEUTRAL,
                itf_trends.iloc[i] == htf_trend,
                ltf_trends.iloc[i] == htf_trend,
            ]) / 3.0,
            htf_timeframe=config.htf,
            itf_timeframe=config.itf,
            ltf_timeframe=config.ltf,
            itf_ob=target_ob if target_ob and target_ob.kind == (OBKind.BULLISH if direction == "long" else OBKind.BEARISH) else None,
            itf_fvg=target_fvg,
        )

        score = score_setup(
            candle_index=i,
            candle_timestamp=timestamps[i],
            direction=direction,
            mtf=mtf,
            ob=target_ob,
            fvg=target_fvg,
            liquidity_levels=eq_levels,
            current_price=current_price,
            swing_high=swing_high,
            swing_low=swing_low,
        )

        if score.total < config.min_score:
            continue

        if direction == "long":
            entry_price = current_price
            stop_loss = entry_zone_low - config.pip_size * 2
            risk = entry_price - stop_loss
            take_profit = entry_price + risk * config.rr_ratio
        else:
            entry_price = current_price
            stop_loss = entry_zone_high + config.pip_size * 2
            risk = stop_loss - entry_price
            take_profit = entry_price - risk * config.rr_ratio

        if risk <= 0:
            continue

        trade_counter += 1
        trade = Trade(
            trade_id=trade_counter,
            direction=TradeDirection.LONG if direction == "long" else TradeDirection.SHORT,
            entry_index=i,
            entry_timestamp=timestamps[i],
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_r=1.0,
            setup_score=score.total,
            setup_notes=score.notes,
        )
        open_trades.append(trade)

        if verbose:
            logger.info(
                f"  T{trade.trade_id} {direction.upper()} @ {entry_price:.5f} "
                f"SL={stop_loss:.5f} TP={take_profit:.5f} Score={score.total:.1f}"
            )

    # Expire remaining open trades at end of data
    final_idx = len(df) - 1
    for trade in open_trades:
        trade.exit_index = final_idx
        trade.exit_timestamp = timestamps[final_idx]
        trade.exit_price = closes[final_idx]
        pnl_pips = (trade.exit_price - trade.entry_price) * (1 if trade.direction == TradeDirection.LONG else -1)
        trade.pnl_r = (pnl_pips / trade.risk_pips) if trade.risk_pips > 0 else 0.0
        trade.status = TradeStatus.EXPIRED
        trades.append(trade)

    return trades, compute_metrics(trades)
