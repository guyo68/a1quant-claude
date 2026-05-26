"""
Backtest performance metrics: win rate, expectancy, Sharpe, max drawdown, by score band.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from .trades import Trade, TradeStatus


@dataclass
class BacktestMetrics:
    total_trades: int
    wins: int
    losses: int
    expired: int
    win_rate: float
    avg_win_r: float
    avg_loss_r: float
    expectancy_r: float  # avg R per trade
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_r: float
    max_drawdown_pct: float
    total_r: float
    avg_mae: float  # average max adverse excursion
    avg_mfe: float  # average max favorable excursion


def compute_metrics(trades: list[Trade]) -> BacktestMetrics:
    """Compute full performance metrics from a list of completed trades."""
    closed = [t for t in trades if t.status != TradeStatus.OPEN and t.pnl_r is not None]

    if not closed:
        return BacktestMetrics(
            total_trades=0, wins=0, losses=0, expired=0,
            win_rate=0.0, avg_win_r=0.0, avg_loss_r=0.0,
            expectancy_r=0.0, profit_factor=0.0, sharpe_ratio=0.0,
            max_drawdown_r=0.0, max_drawdown_pct=0.0, total_r=0.0,
            avg_mae=0.0, avg_mfe=0.0,
        )

    pnls = np.array([t.pnl_r for t in closed])
    wins = [t for t in closed if t.status == TradeStatus.WIN]
    losses = [t for t in closed if t.status == TradeStatus.LOSS]
    expired = [t for t in closed if t.status == TradeStatus.EXPIRED]

    win_pnls = np.array([t.pnl_r for t in wins]) if wins else np.array([0.0])
    loss_pnls = np.array([abs(t.pnl_r) for t in losses]) if losses else np.array([0.0])

    win_rate = len(wins) / len(closed) if closed else 0.0
    avg_win_r = float(np.mean(win_pnls)) if wins else 0.0
    avg_loss_r = float(np.mean(loss_pnls)) if losses else 0.0
    expectancy_r = float(np.mean(pnls))

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe ratio (annualized, assuming ~260 trading days, ~5 trades/day average)
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls)))
    else:
        sharpe = 0.0

    # Max drawdown in R
    equity_curve = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = peak - equity_curve
    max_dd_r = float(np.max(drawdowns))
    total_r = float(equity_curve[-1])
    max_dd_pct = max_dd_r / peak.max() * 100 if peak.max() > 0 else 0.0

    avg_mae = float(np.mean([t.max_adverse_excursion for t in closed]))
    avg_mfe = float(np.mean([t.max_favorable_excursion for t in closed]))

    return BacktestMetrics(
        total_trades=len(closed),
        wins=len(wins),
        losses=len(losses),
        expired=len(expired),
        win_rate=round(win_rate, 4),
        avg_win_r=round(avg_win_r, 3),
        avg_loss_r=round(avg_loss_r, 3),
        expectancy_r=round(expectancy_r, 4),
        profit_factor=round(profit_factor, 3),
        sharpe_ratio=round(sharpe, 3),
        max_drawdown_r=round(max_dd_r, 3),
        max_drawdown_pct=round(max_dd_pct, 2),
        total_r=round(total_r, 3),
        avg_mae=round(avg_mae, 6),
        avg_mfe=round(avg_mfe, 6),
    )


def metrics_by_score_band(trades: list[Trade], bands: list[tuple[float, float]] = None) -> dict[str, BacktestMetrics]:
    """
    Break down metrics by setup score band.
    Default bands: 0–50, 50–60, 60–70, 70–80, 80–100.
    """
    if bands is None:
        bands = [(0, 50), (50, 60), (60, 70), (70, 80), (80, 100)]

    results = {}
    for low, high in bands:
        band_trades = [t for t in trades if low <= t.setup_score < high]
        label = f"{low}-{high}"
        results[label] = compute_metrics(band_trades)
    return results


def print_metrics_report(metrics: BacktestMetrics, title: str = "Backtest Results") -> None:
    """Print a formatted metrics report."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  Total Trades:    {metrics.total_trades}")
    print(f"  Wins/Losses:     {metrics.wins}/{metrics.losses} (Expired: {metrics.expired})")
    print(f"  Win Rate:        {metrics.win_rate:.1%}")
    print(f"  Avg Win (R):     {metrics.avg_win_r:.2f}R")
    print(f"  Avg Loss (R):    {metrics.avg_loss_r:.2f}R")
    print(f"  Expectancy:      {metrics.expectancy_r:.3f}R per trade")
    print(f"  Profit Factor:   {metrics.profit_factor:.2f}")
    print(f"  Sharpe Ratio:    {metrics.sharpe_ratio:.2f}")
    print(f"  Total P&L:       {metrics.total_r:.2f}R")
    print(f"  Max Drawdown:    {metrics.max_drawdown_r:.2f}R ({metrics.max_drawdown_pct:.1f}%)")
    print(f"{'='*50}\n")
