from __future__ import annotations
import numpy as np
import pandas as pd


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    mapping = {"Open":"Open", "High":"High", "Low":"Low", "Close":"Close", "Volume":"Volume",
               "시가":"Open", "고가":"High", "저가":"Low", "종가":"Close", "거래량":"Volume"}
    df = df.rename(columns=mapping)
    if not {"Close", "Volume"}.issubset(df.columns):
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    df = df.loc[~df.index.duplicated(keep="last")].sort_index()
    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close", "Volume"])
    return df.loc[(df.Close > 0) & (df.Volume >= 0)]


def resample(a, n=64):
    a = np.asarray(a, dtype=float)
    if len(a) < 2:
        return np.zeros(n)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(a)), a)


def features(df: pd.DataFrame, n=64):
    close = df.Close.to_numpy(dtype=float)
    volume = df.Volume.to_numpy(dtype=float)
    logprice = np.log(close / close[0])
    logvol = np.log1p(volume)
    # robust, scale-invariant volume pattern
    center = pd.Series(logvol).rolling(20, min_periods=1).median().to_numpy()
    vol_pattern = np.clip(logvol - center, -4, 6)
    ret = np.diff(np.log(close))
    drawdown = close / np.maximum.accumulate(close) - 1
    trend = np.polyfit(np.linspace(0, 1, len(close)), logprice, 1)[0]
    stats = np.array([logprice[-1], drawdown.min(), np.std(ret) * np.sqrt(252), trend])
    return resample(logprice, n), resample(vol_pattern, n), stats


def dtw_distance(a, b, band=10):
    n, m = len(a), len(b)
    prev = np.full(m + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr = np.full(m + 1, np.inf)
        for j in range(max(1, i-band), min(m, i+band)+1):
            curr[j] = abs(a[i-1] - b[j-1]) + min(prev[j], curr[j-1], prev[j-1])
        prev = curr
    return float(prev[m] / max(n, m))


def compare(ref, candidate, weights=(.7, .2, .1)):
    p1, v1, s1 = ref
    p2, v2, s2 = candidate
    # normalized by empirically useful scale constants; scores are not probabilities
    pdist = dtw_distance(p1, p2) / .35
    vdist = dtw_distance(v1, v2) / 1.8
    sdist = float(np.mean(np.abs((s1-s2) / np.array([.6, .5, .9, .7]))))
    parts = np.array([pdist, vdist, sdist])
    dist = float(np.dot(parts, weights))
    return round(100 * np.exp(-dist), 2), round(100*np.exp(-pdist), 2), round(100*np.exp(-vdist), 2), round(100*np.exp(-sdist), 2)


def search_one(ref_features, df, reference_len, min_factor=.75, max_factor=1.25, recent_only=True, asof=None, exclude_breakout=False):
    if asof is not None:
        df = df.loc[:pd.Timestamp(asof)]
    if df.empty:
        return None
    # Last available trading day: candidate is not allowed to look into the future.
    lengths = sorted(set(max(25, int(reference_len * factor)) for factor in np.linspace(min_factor, max_factor, 7)))
    best = None
    for length in lengths:
        if len(df) < length:
            continue
        seg = df.iloc[-length:]
        if exclude_breakout and len(seg) >= 60:
            # Exclude stocks trading far above their recent median or recently surging
            if seg.Close.iloc[-1] / seg.Close.iloc[-60] > 1.8:
                continue
            if seg.Close.iloc[-1] / seg.Close.tail(60).median() > 1.4:
                continue
        result = compare(ref_features, features(seg))
        if best is None or result[0] > best[0]:
            best = (*result, seg.index[0], seg.index[-1], length)
    return best
