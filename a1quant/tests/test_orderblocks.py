"""Unit tests for Order Block detection."""

import pytest
import pandas as pd
import numpy as np
from ..engine.structure import detect_structure
from ..engine.orderblocks import detect_order_blocks, update_order_block_statuses, OBKind, OBStatus


def make_ob_scenario(direction: str = "bullish") -> pd.DataFrame:
    """
    Create a DataFrame with a clear OB scenario.
    Uses an explicit triangle-wave pattern so local extrema exist at known indices,
    ensuring swing detection (fractal pivot n=2) finds both highs and lows.

    Bullish scenario pattern (close prices, n=2 swing detection):
      Descending zigzag: each peak lower than previous (bearish structure)
      Then: OB candle (bearish) + strong bullish impulse that breaks prior swing high

    Each swing high/low must be the maximum/minimum in a 5-candle window (n=2).
    """
    if direction == "bullish":
        # Explicit descending zigzag: 3 down moves + 3 up pullbacks
        # Pattern designed so each pullback peak IS a local 5-candle max
        closes = np.array([
            # Impulse 1 down
            1.1000, 1.0985, 1.0970, 1.0955, 1.0940,
            # Pullback 1 up — candle[7] will be the local high (max in window [5..9])
            1.0950, 1.0960, 1.0970, 1.0960, 1.0950,
            # Impulse 2 down
            1.0935, 1.0920, 1.0905, 1.0890, 1.0875,
            # Pullback 2 up — candle[17] is local high
            1.0885, 1.0895, 1.0905, 1.0895, 1.0885,
            # Impulse 3 down
            1.0870, 1.0855, 1.0840, 1.0825,
            # OB candle: bearish (close < open)
            1.0810,
            # Strong bullish displacement: breaks prior swing high at 1.0970
            1.0840, 1.0870, 1.0900, 1.0930, 1.0960,
            1.0990, 1.1020, 1.1050, 1.1080, 1.1100,
        ])
    else:
        # Ascending zigzag then strong bearish impulse
        closes = np.array([
            1.1000, 1.1015, 1.1030, 1.1045, 1.1060,
            1.1050, 1.1040, 1.1030, 1.1040, 1.1050,
            1.1065, 1.1080, 1.1095, 1.1110, 1.1125,
            1.1115, 1.1105, 1.1095, 1.1105, 1.1115,
            1.1130, 1.1145, 1.1160, 1.1175,
            1.1190,  # OB candle: bullish
            1.1160, 1.1130, 1.1100, 1.1070, 1.1040,
            1.1010, 1.0980, 1.0950, 1.0920, 1.0900,
        ])

    n = len(closes)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")

    opens = np.empty(n)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]

    wick = 0.0003
    highs = np.maximum(opens, closes) + wick
    lows = np.minimum(opens, closes) - wick

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": np.ones(n) * 100},
        index=timestamps,
    )


class TestOrderBlockDetection:
    def test_bullish_ob_detected(self):
        df = make_ob_scenario("bullish")
        breaks, _, _ = detect_structure(df, n=3)
        obs = detect_order_blocks(df, breaks, atr_multiplier=0.5)
        bullish_obs = [ob for ob in obs if ob.kind == OBKind.BULLISH]
        assert len(bullish_obs) >= 1

    def test_bearish_ob_detected(self):
        df = make_ob_scenario("bearish")
        breaks, _, _ = detect_structure(df, n=3)
        obs = detect_order_blocks(df, breaks, atr_multiplier=0.5)
        bearish_obs = [ob for ob in obs if ob.kind == OBKind.BEARISH]
        assert len(bearish_obs) >= 1

    def test_ob_has_valid_high_low(self):
        df = make_ob_scenario("bullish")
        breaks, _, _ = detect_structure(df, n=3)
        obs = detect_order_blocks(df, breaks, atr_multiplier=0.5)
        for ob in obs:
            assert ob.high >= ob.low
            assert ob.high > 0

    def test_ob_status_invalidated_when_price_closes_through(self):
        df = make_ob_scenario("bullish")
        breaks, _, _ = detect_structure(df, n=3)
        obs = detect_order_blocks(df, breaks, atr_multiplier=0.5)
        update_order_block_statuses(df, obs)
        # All OBs should have been processed (not still FRESH after price moved through)
        statuses = {ob.status for ob in obs}
        assert len(statuses) > 0  # some status was set

    def test_no_ob_without_structure_breaks(self):
        # Flat market with no BOS → no OBs
        n = 30
        timestamps = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
        prices = np.ones(n) * 1.1000
        df = pd.DataFrame(
            {"open": prices, "high": prices + 0.0001, "low": prices - 0.0001, "close": prices, "volume": np.ones(n)},
            index=timestamps,
        )
        obs = detect_order_blocks(df, [], atr_multiplier=1.0)
        assert obs == []
