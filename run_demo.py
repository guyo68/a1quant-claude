"""
a1quant Phase 1 demo: SMC detection + backtest on synthetic EUR/USD data.
Usage: python run_demo.py [--days N] [--score MIN_SCORE] [--rr RR_RATIO]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from a1quant.data.dukascopy import load_sample_data
from a1quant.backtest.runner import run_backtest, BacktestConfig
from a1quant.backtest.metrics import print_metrics_report, metrics_by_score_band
from a1quant.engine.structure import detect_structure
from a1quant.engine.orderblocks import detect_order_blocks, update_order_block_statuses
from a1quant.engine.fvg import detect_fvgs, update_fvg_statuses, FVGStatus
from a1quant.engine.liquidity import detect_equal_levels, detect_liquidity_sweeps


def main():
    parser = argparse.ArgumentParser(description="a1quant SMC backtest demo")
    parser.add_argument("--days", type=int, default=30, help="Days of synthetic data (default: 30)")
    parser.add_argument("--score", type=float, default=20.0, help="Minimum setup score to trade (default: 20 for synthetic data; use 60+ on real data)")
    parser.add_argument("--rr", type=float, default=2.0, help="Risk:reward ratio (default: 2.0)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  a1quant — Institutional SMC Trading Platform")
    print(f"  Phase 1: Detection Engine + Backtester")
    print(f"{'='*60}")
    print(f"\n  Instrument : Synthetic EUR/USD (M1)")
    print(f"  Period     : {args.days} days")
    print(f"  Min score  : {args.score}")
    print(f"  R:R target : 1:{args.rr}")

    print(f"\n[1/4] Generating synthetic data...")
    df = load_sample_data(n_days=args.days)
    print(f"      {len(df):,} M1 candles | {df.index[0].date()} → {df.index[-1].date()}")

    print(f"\n[2/4] Running SMC detection...")
    breaks, sw_highs, sw_lows = detect_structure(df, n=5)
    obs = detect_order_blocks(df, breaks, atr_multiplier=1.5)
    update_order_block_statuses(df, obs)
    fvgs = detect_fvgs(df, min_pips=2.0)
    update_fvg_statuses(df, fvgs)
    eq_levels = detect_equal_levels(sw_highs, sw_lows)
    detect_liquidity_sweeps(df, eq_levels)

    bos_count  = sum(1 for b in breaks if b.event.value == "BOS")
    choch_count = sum(1 for b in breaks if b.event.value == "ChoCH")
    fresh_obs   = sum(1 for o in obs if o.status.value == "fresh")
    open_fvgs   = sum(1 for f in fvgs if f.status == FVGStatus.OPEN)
    sweeps      = sum(1 for l in eq_levels if l.status.value == "swept")

    print(f"      Swing points : {len(sw_highs)} highs / {len(sw_lows)} lows")
    print(f"      Structure    : {bos_count} BOS  |  {choch_count} ChoCH")
    print(f"      Order blocks : {len(obs)} total  |  {fresh_obs} fresh")
    print(f"      FVGs         : {len(fvgs)} total  |  {open_fvgs} open")
    print(f"      Liq. sweeps  : {sweeps} of {len(eq_levels)} equal levels taken")

    print(f"\n[3/4] Running backtest (warmup=200 bars)...")
    config = BacktestConfig(
        rr_ratio=args.rr,
        min_score=args.score,
        max_concurrent_trades=1,
    )
    trades, metrics = run_backtest(df, config=config, warmup_bars=200, verbose=False)

    print_metrics_report(metrics, title=f"Full Backtest  (score ≥ {args.score})")

    print(f"\n[4/4] Performance by setup score band:")
    print(f"  {'Band':<10}  {'Trades':>6}  {'Win%':>6}  {'Expectancy':>10}  {'Total R':>8}")
    print(f"  {'-'*48}")
    band_metrics = metrics_by_score_band(trades)
    for band, m in band_metrics.items():
        if m.total_trades == 0:
            continue
        print(f"  {band:<10}  {m.total_trades:>6}  {m.win_rate:>5.1%}  {m.expectancy_r:>+10.3f}R  {m.total_r:>+7.2f}R")

    if trades:
        agrade = [t for t in trades if t.setup_score >= 70]
        print(f"\n  A-grade signals (score ≥ 70): {len(agrade)} of {metrics.total_trades} trades")
        print(f"\n  Sample trades:")
        print(f"  {'#':<4}  {'Dir':<5}  {'Entry':>8}  {'Score':>6}  {'P&L':>7}  {'Status'}")
        print(f"  {'-'*48}")
        for t in trades[:10]:
            pnl = f"{t.pnl_r:+.2f}R" if t.pnl_r is not None else "open"
            print(f"  {t.trade_id:<4}  {t.direction.value:<5}  {t.entry_price:>8.5f}  {t.setup_score:>6.1f}  {pnl:>7}  {t.status.value}")

    print(f"\n  NOTE: Scores on synthetic data are low (~26) because random-walk")
    print(f"  data has no persistent multi-timeframe trend. Real Dukascopy data")
    print(f"  will produce A-grade signals (score ≥ 70). Run:")
    print(f"    python run_demo.py --days 30 --score 60")
    print(f"  after loading EUR/USD Dukascopy data into a1quant/data/processed/.")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
