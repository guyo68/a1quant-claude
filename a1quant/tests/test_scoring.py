"""Unit tests for setup scoring."""

import pytest
import pandas as pd
import numpy as np
from ..engine.scoring import score_setup, AGRADE_THRESHOLD, score_premium_discount, score_ob_quality
from ..engine.mtf import MTFContext
from ..engine.structure import Trend
from ..engine.orderblocks import OrderBlock, OBKind, OBStatus
from ..engine.liquidity import LiquidityLevel, LiquiditySide, LiquidityStatus


def make_perfect_mtf(direction: str = "long") -> MTFContext:
    trend = Trend.BULLISH if direction == "long" else Trend.BEARISH
    return MTFContext(
        htf_trend=trend, itf_trend=trend, ltf_trend=trend,
        htf_aligned=True, itf_aligned=True, ltf_aligned=True,
        alignment_score=1.0,
        htf_timeframe="H4", itf_timeframe="H1", ltf_timeframe="M15",
    )


def make_partial_mtf(direction: str = "long") -> MTFContext:
    trend = Trend.BULLISH if direction == "long" else Trend.BEARISH
    return MTFContext(
        htf_trend=trend, itf_trend=trend, ltf_trend=Trend.NEUTRAL,
        htf_aligned=True, itf_aligned=True, ltf_aligned=False,
        alignment_score=2 / 3,
        htf_timeframe="H4", itf_timeframe="H1", ltf_timeframe="M15",
    )


def make_fresh_ob(direction: str = "long") -> OrderBlock:
    kind = OBKind.BULLISH if direction == "long" else OBKind.BEARISH
    return OrderBlock(
        kind=kind, formed_index=10, formed_timestamp=pd.Timestamp("2023-01-01"),
        high=1.1010, low=1.1000, origin_bos_index=15, status=OBStatus.FRESH,
    )


class TestPremiumDiscount:
    def test_long_in_discount_scores_high(self):
        score = score_premium_discount(price=1.100, swing_high=1.120, swing_low=1.100, direction="long")
        assert score >= 0.9

    def test_long_in_premium_scores_zero(self):
        score = score_premium_discount(price=1.118, swing_high=1.120, swing_low=1.100, direction="long")
        assert score == 0.0

    def test_short_in_premium_scores_high(self):
        score = score_premium_discount(price=1.118, swing_high=1.120, swing_low=1.100, direction="short")
        assert score >= 0.8

    def test_short_in_discount_scores_zero(self):
        score = score_premium_discount(price=1.102, swing_high=1.120, swing_low=1.100, direction="short")
        assert score == 0.0

    def test_equal_high_low_returns_midpoint(self):
        score = score_premium_discount(1.100, 1.100, 1.100, "long")
        assert score == 0.5


class TestOBQuality:
    def test_fresh_ob_scores_one(self):
        ob = make_fresh_ob()
        assert score_ob_quality(ob) == 1.0

    def test_mitigated_ob_scores_less(self):
        ob = make_fresh_ob()
        ob.status = OBStatus.MITIGATED
        assert score_ob_quality(ob) < 1.0

    def test_invalidated_ob_scores_zero(self):
        ob = make_fresh_ob()
        ob.status = OBStatus.INVALIDATED
        assert score_ob_quality(ob) == 0.0


class TestCompositeScore:
    def test_perfect_setup_scores_high(self):
        ts = pd.Timestamp("2023-01-03 08:30", tz="UTC")  # NY session
        mtf = make_perfect_mtf("long")
        ob = make_fresh_ob("long")
        score = score_setup(
            candle_index=50,
            candle_timestamp=ts,
            direction="long",
            mtf=mtf,
            ob=ob,
            fvg=None,
            liquidity_levels=[],
            current_price=1.1005,
            swing_high=1.1200,
            swing_low=1.1000,
        )
        assert score.total >= 60  # good setup even without liquidity sweep

    def test_agrade_threshold(self):
        ts = pd.Timestamp("2023-01-03 08:30", tz="UTC")
        mtf = make_perfect_mtf("long")
        ob = make_fresh_ob("long")
        # Add a liquidity sweep
        sweep_level = LiquidityLevel(
            side=LiquiditySide.SELL_SIDE,
            price=1.0990,
            formed_index=30,
            formed_timestamp=pd.Timestamp("2023-01-02"),
            label="equal_low",
            status=LiquidityStatus.SWEPT,
            sweep_index=45,
            close_back_inside=True,
        )
        score = score_setup(
            candle_index=50,
            candle_timestamp=ts,
            direction="long",
            mtf=mtf,
            ob=ob,
            fvg=None,
            liquidity_levels=[sweep_level],
            current_price=1.1002,
            swing_high=1.1200,
            swing_low=1.1000,
        )
        assert score.is_agrade == (score.total >= AGRADE_THRESHOLD)

    def test_score_components_sum_to_total(self):
        ts = pd.Timestamp("2023-01-03 08:30", tz="UTC")
        mtf = make_partial_mtf("long")
        score = score_setup(
            candle_index=50,
            candle_timestamp=ts,
            direction="long",
            mtf=mtf,
            ob=None,
            fvg=None,
            liquidity_levels=[],
            current_price=1.1050,
            swing_high=1.1200,
            swing_low=1.1000,
        )
        components_sum = (
            score.mtf_score + score.zone_quality_score +
            score.liquidity_sweep_score + score.premium_discount_score +
            score.session_score
        )
        assert abs(components_sum - score.total) < 0.1
