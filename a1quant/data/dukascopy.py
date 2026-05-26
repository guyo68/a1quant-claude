"""
Dukascopy data pipeline.

Dukascopy provides free historical FX tick data and OHLCV CSVs.
Download URL format: https://datafeed.dukascopy.com/datafeed/{INSTRUMENT}/{YEAR}/{MONTH:02d}/{DAY:02d}/BID_candles_min_1.bi5

The .bi5 files are LZMA-compressed binary with 5 floats per record: [time_ms, open, high, low, close, volume].
This module handles: download → decode → clean → save as parquet.
"""

import struct
import lzma
import os
import time
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import requests

logger = logging.getLogger(__name__)

DUKASCOPY_BASE = "https://datafeed.dukascopy.com/datafeed"
RAW_DIR = Path(__file__).parent.parent.parent / "a1quant" / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent.parent / "a1quant" / "data" / "processed"

INSTRUMENTS = {
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
}

PIP_SIZES = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
}

# Price multiplier to convert Dukascopy integer prices to float
PRICE_MULTIPLIERS = {
    "EURUSD": 1e-5,
    "GBPUSD": 1e-5,
    "USDJPY": 1e-3,
}


def _download_bi5(instrument: str, year: int, month: int, day: int, retries: int = 3) -> Optional[bytes]:
    """Download a single day's M1 BID candle file from Dukascopy."""
    # month is 0-indexed in Dukascopy URLs
    url = f"{DUKASCOPY_BASE}/{instrument}/{year}/{month - 1:02d}/{day:02d}/BID_candles_min_1.bi5"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 0:
                return resp.content
            elif resp.status_code == 404:
                return None
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def _decode_bi5(data: bytes, date: pd.Timestamp, price_mult: float) -> pd.DataFrame:
    """
    Decode a .bi5 binary file into a DataFrame.
    Each record is 5 big-endian values: time_ms (uint32), open, high, low, close (uint32), volume (float32).
    Total: 4 + 4 + 4 + 4 + 4 + 4 = 24 bytes per record.
    """
    try:
        raw = lzma.decompress(data)
    except lzma.LZMAError:
        return pd.DataFrame()

    record_size = 24
    n_records = len(raw) // record_size
    if n_records == 0:
        return pd.DataFrame()

    records = []
    for i in range(n_records):
        chunk = raw[i * record_size : (i + 1) * record_size]
        time_ms, o, h, l, c, v = struct.unpack(">IIIIIf", chunk)
        ts = date + pd.Timedelta(milliseconds=int(time_ms))
        records.append({
            "timestamp": ts,
            "open": o * price_mult,
            "high": h * price_mult,
            "low": l * price_mult,
            "close": c * price_mult,
            "volume": v,
        })

    df = pd.DataFrame(records)
    df.set_index("timestamp", inplace=True)
    return df


def download_instrument(
    instrument: str,
    start_date: str,
    end_date: str,
    raw_dir: Optional[Path] = None,
    delay_seconds: float = 0.3,
) -> None:
    """
    Download all daily .bi5 files for an instrument between start_date and end_date.
    Files are cached locally — already-downloaded days are skipped.
    """
    raw_dir = raw_dir or RAW_DIR / instrument
    raw_dir.mkdir(parents=True, exist_ok=True)
    price_mult = PRICE_MULTIPLIERS.get(instrument, 1e-5)

    dates = pd.date_range(start_date, end_date, freq="D")
    downloaded = 0
    skipped = 0

    for date in dates:
        fname = raw_dir / f"{date.strftime('%Y%m%d')}.bi5"
        if fname.exists():
            skipped += 1
            continue

        data = _download_bi5(instrument, date.year, date.month, date.day)
        if data:
            fname.write_bytes(data)
            downloaded += 1
            logger.info(f"Downloaded {instrument} {date.date()}")
        else:
            logger.debug(f"No data for {instrument} {date.date()} (weekend/holiday)")

        time.sleep(delay_seconds)

    logger.info(f"{instrument}: {downloaded} downloaded, {skipped} cached, {len(dates) - downloaded - skipped} missing")


def load_raw_to_dataframe(instrument: str, raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load all cached .bi5 files for an instrument and concatenate into a single DataFrame."""
    raw_dir = raw_dir or RAW_DIR / instrument
    price_mult = PRICE_MULTIPLIERS.get(instrument, 1e-5)

    frames = []
    for fpath in sorted(raw_dir.glob("*.bi5")):
        date_str = fpath.stem  # e.g. "20200103"
        try:
            date = pd.Timestamp(date_str, tz="UTC")
        except Exception:
            continue
        data = fpath.read_bytes()
        df = _decode_bi5(data, date, price_mult)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


def process_and_save(
    instrument: str,
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
) -> Path:
    """
    Load raw .bi5 files, clean, and save as parquet.
    Returns the parquet file path.
    """
    processed_dir = processed_dir or PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"{instrument}_M1.parquet"

    df = load_raw_to_dataframe(instrument, raw_dir)
    if df.empty:
        raise ValueError(f"No data found for {instrument}")

    # Remove rows with zero prices (data errors)
    df = df[(df["open"] > 0) & (df["close"] > 0)]
    # Forward-fill gaps up to 5 minutes (weekend gaps handled by not filling)
    df = df.asfreq("1min").ffill(limit=5)
    df.dropna(inplace=True)

    df.to_parquet(out_path)
    logger.info(f"Saved {len(df):,} rows to {out_path}")
    return out_path


def load_parquet(instrument: str, timeframe: str = "M1", processed_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load processed parquet data for an instrument."""
    processed_dir = processed_dir or PROCESSED_DIR
    path = processed_dir / f"{instrument}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No processed data at {path}. Run process_and_save() first.")
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def load_sample_data(instrument: str = "EURUSD", n_days: int = 30) -> pd.DataFrame:
    """
    Generate synthetic M1 OHLCV data for testing when real data is unavailable.
    Uses a random walk with realistic FX volatility.
    """
    np.random.seed(42)
    n_minutes = n_days * 24 * 60
    timestamps = pd.date_range("2023-01-01", periods=n_minutes, freq="1min", tz="UTC")

    # EUR/USD realistic parameters
    start_price = 1.0850
    volatility = 0.0001  # per minute
    returns = np.random.normal(0, volatility, n_minutes)
    prices = start_price * np.exp(np.cumsum(returns))

    # Build OHLCV with realistic candle structure
    noise = np.random.uniform(0.00005, 0.00020, (n_minutes, 2))
    opens = prices * (1 + np.random.normal(0, 0.00005, n_minutes))
    highs = np.maximum(opens, prices) + noise[:, 0]
    lows = np.minimum(opens, prices) - noise[:, 1]
    closes = prices

    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": np.random.uniform(100, 1000, n_minutes)},
        index=timestamps,
    )
    return df
