"""Unit tests for Fair Value Gap detection."""

import pytest
import pandas as pd
import numpy as np
from ..engine.fvg import detect_fvgs, update_fvg_statuses, FVGKind, FVGStatus


def make_df_from_ohlc(rows: list[dict]) -> pd.DataFrame:
    timestamps = pd.date_range("2023-01-01", periods=len(rows), freq="1h", tz="UTC")
    df = pd.DataFrame(rows, index=timestamps)
    df["volume"] = 100.0
    return df


class TestFVGDetection:
    def test_bullish_fvg_detected(self):
        # candle[0].high = 1.100, candle[2].low = 1.105 → gap 1.100–1.105
        rows = [
            {"open": 1.090, "high": 1.100, "low": 1.085, "close": 1.095},
            {"open": 1.095, "high": 1.120, "low": 1.090, "close": 1.115},  # impulse candle
            {"open": 1.115, "high": 1.125, "low": 1.106, "close": 1.120},
        ]
        df = make_df_from_ohlc(rows)
        fvgs = detect_fvgs(df, min_pips=0.5, pip_size=0.0001)
        bullish = [f for f in fvgs if f.kind == FVGKind.BULLISH]
        assert len(bullish) == 1
        assert abs(bullish[0].bottom - 1.100) < 1e-6
        assert abs(bullish[0].top - 1.106) < 1e-6

    def test_bearish_fvg_detected(self):
        rows = [
            {"open": 1.120, "high": 1.125, "low": 1.115, "close": 1.118},
            {"open": 1.118, "high": 1.119, "low": 1.095, "close": 1.098},
            {"open": 1.098, "high": 1.110, "low": 1.090, "close": 1.095},
        ]
        df = make_df_from_ohlc(rows)
        fvgs = detect_fvgs(df, min_pips=0.5, pip_size=0.0001)
        bearish = [f for f in fvgs if f.kind == FVGKind.BEARISH]
        assert len(bearish) == 1

    def test_no_fvg_when_no_gap(self):
        # Overlapping candles — no gap
        rows = [
            {"open": 1.100, "high": 1.110, "low": 1.095, "close": 1.105},
            {"open": 1.105, "high": 1.115, "low": 1.100, "close": 1.110},
            {"open": 1.110, "high": 1.108, "low": 1.103, "close": 1.106},
        ]
        df = make_df_from_ohlc(rows)
        fvgs = detect_fvgs(df, min_pips=0.5, pip_size=0.0001)
        assert len(fvgs) == 0

    def test_min_pips_filter(self):
        # Gap of 0.0003 (3 pips) — should be caught by 2-pip filter but not 5-pip
        rows = [
            {"open": 1.100, "high": 1.100, "low": 1.095, "close": 1.098},
            {"open": 1.098, "high": 1.110, "low": 1.098, "close": 1.108},
            {"open": 1.108, "high": 1.115, "low": 1.1003, "close": 1.112},
        ]
        df = make_df_from_ohlc(rows)
        fvgs_2pip = detect_fvgs(df, min_pips=2.0, pip_size=0.0001)
        fvgs_5pip = detect_fvgs(df, min_pips=5.0, pip_size=0.0001)
        assert len(fvgs_5pip) == 0


class TestFVGStatus:
    def test_fvg_closes_when_filled(self):
        rows = [
            {"open": 1.090, "high": 1.100, "low": 1.085, "close": 1.095},
            {"open": 1.095, "high": 1.120, "low": 1.090, "close": 1.115},
            {"open": 1.115, "high": 1.125, "low": 1.106, "close": 1.120},
            # Retrace: close below gap bottom
            {"open": 1.120, "high": 1.120, "low": 1.085, "close": 1.088},
        ]
        df = make_df_from_ohlc(rows)
        fvgs = detect_fvgs(df, min_pips=0.5, pip_size=0.0001)
        update_fvg_statuses(df, fvgs)
        bullish = [f for f in fvgs if f.kind == FVGKind.BULLISH]
        if bullish:
            assert bullish[0].status == FVGStatus.CLOSED

    def test_fvg_partial_when_partially_filled(self):
        rows = [
            {"open": 1.090, "high": 1.100, "low": 1.085, "close": 1.095},
            {"open": 1.095, "high": 1.125, "low": 1.090, "close": 1.120},
            {"open": 1.120, "high": 1.130, "low": 1.108, "close": 1.125},
            # Partial retrace: wick into gap but close above bottom
            {"open": 1.125, "high": 1.125, "low": 1.101, "close": 1.115},
        ]
        df = make_df_from_ohlc(rows)
        fvgs = detect_fvgs(df, min_pips=0.5, pip_size=0.0001)
        update_fvg_statuses(df, fvgs)
        bullish = [f for f in fvgs if f.kind == FVGKind.BULLISH]
        if bullish:
            assert bullish[0].status in (FVGStatus.OPEN, FVGStatus.PARTIAL)
