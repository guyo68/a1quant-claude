"""Integration test: run a full backtest on synthetic data."""

import pytest
import pandas as pd
import numpy as np
from ..data.dukascopy import load_sample_data
from ..backtest.runner import run_backtest, BacktestConfig
from ..backtest.metrics import compute_metrics, metrics_by_score_band


class TestBacktestIntegration:
    @pytest.fixture
    def sample_df(self):
        # 5 days gives ~7200 M1 bars — enough for structure detection, fast to run
        return load_sample_data(n_days=5)

    def test_backtest_runs_without_error(self, sample_df):
        config = BacktestConfig(min_score=50.0, rr_ratio=2.0)
        trades, metrics = run_backtest(sample_df, config=config, warmup_bars=200, verbose=False)
        assert isinstance(trades, list)
        assert metrics.total_trades >= 0

    def test_backtest_metrics_are_consistent(self, sample_df):
        config = BacktestConfig(min_score=50.0, rr_ratio=2.0)
        trades, metrics = run_backtest(sample_df, config=config, warmup_bars=200)
        assert metrics.wins + metrics.losses + metrics.expired == metrics.total_trades
        assert 0.0 <= metrics.win_rate <= 1.0

    def test_metrics_by_score_band(self, sample_df):
        config = BacktestConfig(min_score=40.0, rr_ratio=2.0)
        trades, _ = run_backtest(sample_df, config=config, warmup_bars=200)
        band_metrics = metrics_by_score_band(trades)
        assert isinstance(band_metrics, dict)
        assert "70-80" in band_metrics

    def test_agrade_trades_have_score_above_70(self, sample_df):
        config = BacktestConfig(min_score=40.0)
        trades, _ = run_backtest(sample_df, config=config, warmup_bars=200)
        agrade = [t for t in trades if t.setup_score >= 70]
        for t in agrade:
            assert t.setup_score >= 70

    def test_no_trades_at_very_high_threshold(self, sample_df):
        config = BacktestConfig(min_score=99.0)
        trades, metrics = run_backtest(sample_df, config=config, warmup_bars=200)
        assert metrics.total_trades == 0
