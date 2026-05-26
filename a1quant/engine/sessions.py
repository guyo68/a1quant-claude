"""
Session timing context for scoring: determine which session a candle belongs to.
"""

import pandas as pd
from enum import Enum


class Session(Enum):
    ASIA = "asia"
    LONDON = "london"
    NEW_YORK = "new_york"
    OFF_SESSION = "off_session"


# Session hours in EST (UTC-5 / UTC-4 during DST — approximate)
SESSION_HOURS_EST = {
    Session.ASIA: (20, 24),       # 20:00–00:00 (wraps midnight)
    Session.LONDON: (2, 5),       # 02:00–05:00
    Session.NEW_YORK: (7, 10),    # 07:00–10:00
}

# Relative session quality for scoring (NY and London are highest quality)
SESSION_SCORE_WEIGHT = {
    Session.NEW_YORK: 1.0,
    Session.LONDON: 0.8,
    Session.ASIA: 0.3,
    Session.OFF_SESSION: 0.1,
}


def get_session(timestamp: pd.Timestamp, tz: str = "America/New_York") -> Session:
    """Return the trading session for a given UTC timestamp."""
    if timestamp.tzinfo is None:
        ts_local = timestamp.tz_localize("UTC").tz_convert(tz)
    else:
        ts_local = timestamp.tz_convert(tz)

    hour = ts_local.hour

    # Asia wraps midnight: 20–23 or 0
    if hour >= 20 or hour < 0:
        return Session.ASIA
    if 2 <= hour < 5:
        return Session.LONDON
    if 7 <= hour < 10:
        return Session.NEW_YORK
    return Session.OFF_SESSION


def get_session_weight(timestamp: pd.Timestamp, tz: str = "America/New_York") -> float:
    """Return the session quality weight (0–1) for scoring purposes."""
    return SESSION_SCORE_WEIGHT[get_session(timestamp, tz)]


def classify_dataframe_sessions(df: pd.DataFrame, tz: str = "America/New_York") -> pd.Series:
    """Add a session column to a DataFrame indexed by timestamp."""
    return df.index.to_series().apply(lambda ts: get_session(ts, tz).value)
