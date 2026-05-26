"""
Backtest runner: orchestrates the full pipeline on historical data.

For each bar (candle-by-candle replay):
  1. Run structure detection up to current bar
  2. Detect OBs and FVGs, update statuses
  3. Map liquidity levels
  4. Compute MTF alignment
  5. Score any active setups
  6. If score ≥ threshold, generate a trade entry
  7. Simulate trade forward, record outcome
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
from ..engine.mtf import get_mtf_alignment, resample_ohlcv
from ..engine.scoring import score_setup, AGRADE_THRESHOLD
from .trades import Trade, TradeDirection, simulate_trade
from .metrics import compute_metrics, metrics_by_score_band, BacktestMetrics, print_metrics_report

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    rr_ratio: float = 2.0          # take profit = entry + rr_ratio × risk
    min_score: float = 60.0        # minimum score to enter a trade
    max_concurrent_trades: int = 1  # max open trades at once
    swing_n: int = 5               # fractal pivot window
    atr_multiplier: float = 1.5    # OB displacement threshold
    atr_period: int = 14
    fvg_min_pips: float = 2.0      # minimum FVG size
    pip_size: float = 0.0001
    max_bars_in_trade: int = 500   # bars before trade expires
    htf: str = "H4"
    itf: str = "H1"
    ltf: str = "M15"
    # How far back to look for liquidity sweeps (bars)
    sweep_lookback: int = 20
    # Only trade during specific sessions: None = all sessions
    session_filter: Optional[list[str]] = None


def run_backtest(
    df: pd.DataFrame,
    config: BacktestConfig = None,
    warmup_bars: int = 500,
    verbose: bool = False,
) -> tuple[list[Trade], BacktestMetrics]:
    """
    Run a full bar-by-bar backtest on M1 OHLCV data.
    Returns (trades, metrics).

    warmup_bars: number of initial bars to skip (needed for indicator warmup).
    """
    if config is None:
        config = BacktestConfig()

    trades: list[Trade] = []
    trade_counter = 0
    open_trades: list[Trade] = []

    # Pre-compute full structural analysis once (not recalculated per bar for speed)
    # For walk-forward integrity, we use as_of slicing in MTF but pre-compute
    # structure on a rolling basis every N bars to balance accuracy and speed.
    structure_update_interval = 50  # recompute structure every 50 bars

    cached_obs = []
    cached_fvgs = []
    cached_liq_levels: list[LiquidityLevel] = []
    cached_mtf = None
    cached_swing_high = float("nan")
    cached_swing_low = float("nan")
    last_structure_update = 0
    last_mtf_update = 0

    closes = df["close"].values
    opens = df["open"].values
    timestamps = df.index

    logger.info(f"Starting backtest: {len(df):,} bars, warmup={warmup_bars}")

    for i in range(warmup_bars, len(df) - 1):
        df_slice = df.iloc[: i + 1]

        # Recompute structural elements and MTF periodically (not per-bar — O(n²) otherwise)
        if i - last_structure_update >= structure_update_interval or i == warmup_bars:
            try:
                sb, sw_highs, sw_lows = detect_structure(df_slice, n=config.swing_n)
                cached_obs = detect_order_blocks(df_slice, sb, config.atr_multiplier, config.atr_period)
                update_order_block_statuses(df_slice, cached_obs)
                cached_fvgs = detect_fvgs(df_slice, config.fvg_min_pips, config.pip_size)
                update_fvg_statuses(df_slice, cached_fvgs)
                eq_levels = detect_equal_levels(sw_highs, sw_lows, pip_size=config.pip_size)
                detect_liquidity_sweeps(df_slice, eq_levels, pip_size=config.pip_size)
                cached_liq_levels = eq_levels
                df_htf_cache = resample_ohlcv(df_slice, config.htf)
                if len(df_htf_cache) >= 10:
                    cached_swing_high = float(df_htf_cache["high"].rolling(20, min_periods=1).max().iloc[-1])
                    cached_swing_low = float(df_htf_cache["low"].rolling(20, min_periods=1).min().iloc[-1])
                last_structure_update = i
            except Exception as e:
                logger.debug(f"Structure update failed at bar {i}: {e}")
                continue

        if i - last_mtf_update >= structure_update_interval or cached_mtf is None:
            try:
                cached_mtf = get_mtf_alignment(
                    df_slice, timestamps[i],
                    htf=config.htf, itf=config.itf, ltf=config.ltf,
                    swing_n=config.swing_n, atr_multiplier=config.atr_multiplier,
                )
                last_mtf_update = i
            except Exception:
                cached_mtf = None

        # Resolve any open trades
        still_open = []
        for trade in open_trades:
            updated = simulate_trade(trade, df, max_bars=config.max_bars_in_trade)
            if updated.status.value != "open":
                trades.append(updated)
                if verbose:
                    logger.info(f"Trade {updated.trade_id} closed: {updated.status.value} {updated.pnl_r:.2f}R")
            else:
                still_open.append(updated)
        open_trades = still_open

        # Skip if at max concurrent positions
        if len(open_trades) >= config.max_concurrent_trades:
            continue

        if cached_mtf is None:
            continue

        mtf = cached_mtf

        if mtf.htf_trend == Trend.NEUTRAL:
            continue

        direction = "long" if mtf.htf_trend == Trend.BULLISH else "short"
        current_price = closes[i]

        # Find best active OB or FVG for entry
        active_obs = get_fresh_order_blocks(cached_obs, i)
        active_fvgs = get_open_fvgs(cached_fvgs, i)

        target_ob = None
        target_fvg = None
        entry_zone_high = None
        entry_zone_low = None

        # Find an OB/FVG that price is currently touching
        for ob in reversed(active_obs):
            if direction == "long" and ob.kind == OBKind.BULLISH and ob.contains(current_price):
                target_ob = ob
                entry_zone_high = ob.high
                entry_zone_low = ob.low
                break
            elif direction == "short" and ob.kind == OBKind.BEARISH and ob.contains(current_price):
                target_ob = ob
                entry_zone_high = ob.high
                entry_zone_low = ob.low
                break

        if target_ob is None:
            for fvg in reversed(active_fvgs):
                if direction == "long" and fvg.kind == FVGKind.BULLISH and fvg.contains(current_price):
                    target_fvg = fvg
                    entry_zone_high = fvg.top
                    entry_zone_low = fvg.bottom
                    break
                elif direction == "short" and fvg.kind == FVGKind.BEARISH and fvg.contains(current_price):
                    target_fvg = fvg
                    entry_zone_high = fvg.top
                    entry_zone_low = fvg.bottom
                    break

        if target_ob is None and target_fvg is None:
            continue

        if math.isnan(cached_swing_high) or math.isnan(cached_swing_low):
            continue
        swing_high = cached_swing_high
        swing_low = cached_swing_low

        score = score_setup(
            candle_index=i,
            candle_timestamp=timestamps[i],
            direction=direction,
            mtf=mtf,
            ob=target_ob,
            fvg=target_fvg,
            liquidity_levels=cached_liq_levels,
            current_price=current_price,
            swing_high=swing_high,
            swing_low=swing_low,
        )

        if score.total < config.min_score:
            continue

        # Build trade
        if direction == "long":
            entry_price = current_price
            stop_loss = entry_zone_low - config.pip_size * 2  # 2 pip buffer below zone
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
                f"Trade {trade.trade_id} {direction.upper()} @ {entry_price:.5f} "
                f"SL={stop_loss:.5f} TP={take_profit:.5f} Score={score.total:.1f}"
            )

    # Close any remaining open trades at end of data
    for trade in open_trades:
        final_idx = len(df) - 1
        trade.exit_index = final_idx
        trade.exit_timestamp = timestamps[final_idx]
        trade.exit_price = closes[final_idx]
        pnl_pips = (trade.exit_price - trade.entry_price) * (1 if trade.direction == TradeDirection.LONG else -1)
        trade.pnl_r = (pnl_pips / trade.risk_pips) if trade.risk_pips > 0 else 0.0
        from .trades import TradeStatus
        trade.status = TradeStatus.EXPIRED
        trades.append(trade)

    metrics = compute_metrics(trades)
    return trades, metrics
