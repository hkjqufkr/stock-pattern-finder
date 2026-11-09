from __future__ import annotations
import pandas as pd
import streamlit as st
from engine import clean_ohlcv

@st.cache_data(ttl=86400, show_spinner=False)
def listing():
    import FinanceDataReader as fdr
    frames = []
    for market in ('KOSPI','KOSDAQ'):
        try:
            df = fdr.StockListing(market)
            sym = 'Code' if 'Code' in df.columns else 'Symbol'
            if sym not in df.columns or 'Name' not in df.columns:
                continue
            out = df[[sym,'Name']].copy()
            out.columns = ['Code','Name']
            out['Code'] = out['Code'].astype(str).str.zfill(6)
            out['Market'] = market
            frames.append(out)
        except Exception:
            pass
    if not frames:
        raise RuntimeError('상장목록을 불러오지 못했습니다. 네트워크와 데이터 제공처 상태를 확인하세요.')
    return pd.concat(frames, ignore_index=True).drop_duplicates('Code')

@st.cache_data(ttl=43200, show_spinner=False, max_entries=3500)
def history(code, start, end):
    import FinanceDataReader as fdr
    try:
        df = clean_ohlcv(fdr.DataReader(str(code).zfill(6), start, end))
        if len(df) > 10:
            return df
    except Exception:
        pass
    try:
        from pykrx import stock
        df = stock.get_market_ohlcv_by_date(pd.Timestamp(start).strftime('%Y%m%d'),
                                            pd.Timestamp(end).strftime('%Y%m%d'), str(code).zfill(6))
        return clean_ohlcv(df)
    except Exception:
        return pd.DataFrame()
