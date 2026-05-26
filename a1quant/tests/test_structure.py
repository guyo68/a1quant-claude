"""Unit tests for swing detection and structure analysis."""

import pytest
import pandas as pd
import numpy as np
from ..engine.structure import detect_swings, detect_structure, StructureEvent, Trend


def make_df(prices: list[float]) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame from a list of close prices."""
    n = len(prices)
    prices = np.array(prices)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    noise = 0.0002
    return pd.DataFrame(
        {
            "open": prices * (1 + np.random.uniform(-noise, 0, n)),
            "high": prices + np.abs(np.random.normal(0, noise * 2, n)),
            "low": prices - np.abs(np.random.normal(0, noise * 2, n)),
            "close": prices,
            "volume": np.ones(n) * 100,
        },
        index=timestamps,
    )


def make_trending_df(direction: str = "up", n: int = 100, n_swings: int = 4) -> pd.DataFrame:
    """
    Create a DataFrame with clear HH/HL (up) or LL/LH (down) structure.
    """
    np.random.seed(0)
    prices = []
    base = 1.1000
    swing_size = 0.0050
    noise = 0.0005

    for s in range(n_swings):
        # Impulse
        if direction == "up":
            impulse = np.linspace(base, base + swing_size, n // (n_swings * 2))
            pullback_end = base + swing_size * 0.4
            pullback = np.linspace(base + swing_size, pullback_end, n // (n_swings * 2))
            base = pullback_end
        else:
            impulse = np.linspace(base, base - swing_size, n // (n_swings * 2))
            pullback_end = base - swing_size * 0.4
            pullback = np.linspace(base - swing_size, pullback_end, n // (n_swings * 2))
            base = pullback_end
        prices.extend(impulse)
        prices.extend(pullback)

    prices = np.array(prices[:n]) + np.random.normal(0, noise, min(len(prices), n))
    timestamps = pd.date_range("2023-01-01", periods=len(prices), freq="1h", tz="UTC")
    noise_arr = np.abs(np.random.normal(0, noise, len(prices)))
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + noise_arr,
            "low": prices - noise_arr,
            "close": prices,
            "volume": np.ones(len(prices)) * 100,
        },
        index=timestamps,
    )


class TestSwingDetection:
    def test_detects_obvious_swing_high(self):
        # V-shape with clear peak in middle
        prices = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0]
        df = make_df(prices)
        highs, lows = detect_swings(df, n=2)
        # Should find a swing high near index 5
        assert any(abs(h.price - 1.5) < 0.01 for h in highs)

    def test_detects_obvious_swing_low(self):
        prices = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        df = make_df(prices)
        highs, lows = detect_swings(df, n=2)
        assert any(abs(l.price - 1.0) < 0.01 for l in lows)

    def test_n_parameter_controls_sensitivity(self):
        # Noisy data: larger N should find fewer swings
        np.random.seed(42)
        prices = 1.0 + np.random.normal(0, 0.01, 200)
        df = make_df(prices.tolist())
        highs_n2, _ = detect_swings(df, n=2)
        highs_n10, _ = detect_swings(df, n=10)
        assert len(highs_n2) > len(highs_n10)

    def test_returns_lists(self):
        df = make_df([1.0, 1.1, 1.2, 1.1, 1.0])
        highs, lows = detect_swings(df, n=1)
        assert isinstance(highs, list)
        assert isinstance(lows, list)


class TestStructureDetection:
    def test_uptrend_produces_bos(self):
        df = make_trending_df("up", n=120, n_swings=5)
        events, _, _ = detect_structure(df, n=3)
        bos_events = [e for e in events if e.event == StructureEvent.BOS]
        assert len(bos_events) >= 1

    def test_downtrend_produces_bos(self):
        df = make_trending_df("down", n=120, n_swings=5)
        events, _, _ = detect_structure(df, n=3)
        bos_events = [e for e in events if e.event == StructureEvent.BOS]
        assert len(bos_events) >= 1

    def test_reversal_produces_choch(self):
        # Up then down: first BOS up, then ChoCH when direction reverses
        up = make_trending_df("up", n=80, n_swings=3)
        down = make_trending_df("down", n=80, n_swings=3)
        # Shift down prices to continue from where up left off
        offset = up["close"].iloc[-1] - down["close"].iloc[0]
        for col in ["open", "high", "low", "close"]:
            down[col] = down[col] + offset
        down.index = pd.date_range(up.index[-1] + pd.Timedelta(hours=1), periods=len(down), freq="1h", tz="UTC")
        df = pd.concat([up, down])
        events, _, _ = detect_structure(df, n=3)
        choch_events = [e for e in events if e.event == StructureEvent.CHOCH]
        assert len(choch_events) >= 1

    def test_structure_events_are_chronological(self):
        df = make_trending_df("up", n=200, n_swings=6)
        events, _, _ = detect_structure(df, n=3)
        indices = [e.index for e in events]
        assert indices == sorted(indices)

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC"),
        )
        events, highs, lows = detect_structure(df, n=3)
        assert events == []
        assert highs == []
        assert lows == []
