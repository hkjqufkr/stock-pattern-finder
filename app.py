import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data_sources import listing, history
from engine import features, search_one, resample

st.set_page_config(page_title='Stock Pattern Finder', page_icon='📈', layout='wide')
st.title('📈 Stock Pattern Finder')
st.caption('국내 주식 일봉 패턴 유사도 검색 · 가격 70% / 거래량 20% / 추세·변동성 10% · Flex DTW')

try:
    stocks = listing()
except Exception as e:
    st.error(str(e)); st.stop()

with st.sidebar:
    st.header('기준 패턴')
    query = st.text_input('종목명 또는 코드', '현대약품')
    matches = stocks[stocks.Name.str.contains(query, case=False, regex=False, na=False) | stocks.Code.str.contains(query, regex=False)]
    if matches.empty:
        st.warning('종목을 찾지 못했습니다.'); st.stop()
    options = {f'{r.Name} ({r.Code}) · {r.Market}':r.Code for r in matches.itertuples(index=False)}
    selected = st.selectbox('기준 종목', list(options))
    code = options[selected]
    start = st.date_input('기준 시작일', dt.date(2024,7,1))
    end = st.date_input('기준 종료일', dt.date(2025,11,30))
    st.header('검색 조건')
    market = st.selectbox('검색 시장', ['전체','KOSPI','KOSDAQ'])
    max_stocks = st.number_input('검색할 종목 수 (0=전체)', min_value=0, max_value=4000, value=100, step=50)
    min_factor,max_factor = st.slider('Flex 길이 배율', .5, 1.5, (.75,1.25), .05)
    asof = st.date_input('검색 기준일 (이후 데이터 미사용)', dt.date.today())
    exclude_breakout = st.checkbox('최근 급등 종목 제외', value=True)
    topn = st.slider('결과 수', 5, 50, 15)
    st.info('전체 종목 검색은 데이터 제공처 속도와 호출 제한에 따라 오래 걸릴 수 있습니다.')

if start >= end:
    st.error('종료일이 시작일보다 늦어야 합니다.'); st.stop()
ref_all = history(code, str(start - dt.timedelta(days=30)), str(end + dt.timedelta(days=1)))
if ref_all.empty:
    st.error('기준 종목의 일봉을 가져오지 못했습니다.'); st.stop()
ref = ref_all.loc[str(start):str(end)].copy()
if len(ref) < 30:
    st.error('기준 구간은 30거래일 이상이어야 합니다.'); st.stop()

st.subheader('① 기준 차트와 구간 선택')
fig = go.Figure()
fig.add_trace(go.Scatter(x=ref_all.index, y=ref_all.Close, name='종가', line=dict(color='#0f766e')))
fig.add_vrect(x0=str(start), x1=str(end), fillcolor='#0f766e', opacity=.12, line_width=0)
fig.update_layout(height=330, xaxis_title='날짜', yaxis_title='종가 (원)', dragmode='select', margin=dict(l=20,r=20,t=10,b=20))
event = st.plotly_chart(fig, use_container_width=True, on_select='rerun', selection_mode='box', key='reference_chart')
# Plotly selection box can capture a range of points; persist chosen dates via session state.
try:
    pts = event.selection.points
    xs = sorted(pd.to_datetime(p['x']) for p in pts if 'x' in p)
    if len(xs) >= 30:
        if st.button(f'선택한 구간 사용: {xs[0].date()} ~ {xs[-1].date()}'):
            st.session_state['selected_range'] = (xs[0], xs[-1])
except (AttributeError, TypeError, ValueError, KeyError):
    pass
if 'selected_range' in st.session_state:
    a,b = st.session_state['selected_range']
    if st.checkbox(f'차트 드래그 구간 적용 ({a.date()} ~ {b.date()})', True):
        ref = ref_all.loc[a:b]
st.caption(f'기준: {ref.index[0].date()} ~ {ref.index[-1].date()} · {len(ref)}거래일. 드래그 구간이 적용되지 않으면 사이드바 날짜를 사용하세요.')

st.subheader('② 전체 종목 패턴 검색')
if st.button('🔎 유사 차트 검색 시작', type='primary', use_container_width=True):
    subset = stocks if market == '전체' else stocks[stocks.Market == market]
    subset = subset[subset.Code != code].reset_index(drop=True)
    if max_stocks:
        subset = subset.head(int(max_stocks))
    ref_feat = features(ref)
    # Buffer based on longest possible candidate window, including non-trading days.
    days_back = int(len(ref) * max_factor * 2.1) + 60
    first_day = pd.Timestamp(asof) - pd.Timedelta(days=days_back)
    rows = []
    progress = st.progress(0)
    status = st.empty()
    for i, r in enumerate(subset.itertuples(index=False)):
        try:
            df = history(r.Code, first_day.strftime('%Y-%m-%d'), (pd.Timestamp(asof)+pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
            result = search_one(ref_feat, df, len(ref), min_factor, max_factor, asof=asof, exclude_breakout=exclude_breakout)
            if result:
                score, ps, vs, ts, a, b, length = result
                rows.append({'종목':r.Name,'코드':r.Code,'시장':r.Market,'유사도':score,'가격점수':ps,'거래량점수':vs,'추세점수':ts,'시작':a,'종료':b,'거래일':length})
        except Exception:
            pass
        if i % 5 == 0 or i == len(subset)-1:
            progress.progress((i+1)/max(1,len(subset)))
            status.caption(f'{i+1:,} / {len(subset):,} 종목 분석 · 후보 {len(rows):,}개')
    st.session_state['results'] = pd.DataFrame(rows).sort_values('유사도',ascending=False).head(topn).reset_index(drop=True) if rows else pd.DataFrame()
    st.session_state['ref_dates'] = (ref.index[0], ref.index[-1])

if 'results' in st.session_state:
    results = st.session_state['results']
    if results.empty:
        st.warning('조건에 맞는 종목이 없습니다. 검색 수를 늘리거나 최근 급등 제외 필터를 해제해 보세요.')
    else:
        st.dataframe(results.assign(시작=results['시작'].dt.strftime('%Y-%m-%d'), 종료=results['종료'].dt.strftime('%Y-%m-%d')), use_container_width=True, hide_index=True)
        st.download_button('CSV 다운로드', results.to_csv(index=False).encode('utf-8-sig'), 'pattern_matches.csv', 'text/csv')
        st.subheader('③ 겹쳐보기')
        names = [f'{r.종목} ({r.코드}) · {r.유사도}점' for r in results.itertuples(index=False)]
        chosen = st.selectbox('비교 종목', names)
        r = results.iloc[names.index(chosen)]
        candidate = history(r['코드'], str(r['시작'].date()), str((r['종료']+pd.Timedelta(days=1)).date()))
        candidate = candidate.loc[r['시작']:r['종료']]
        if len(candidate) >= 2:
            c1,c2 = st.columns(2)
            n=100
            with c1:
                overlay = go.Figure()
                overlay.add_trace(go.Scatter(x=np.linspace(0,100,n), y=100*np.exp(resample(np.log(ref.Close/ref.Close.iloc[0]),n)), name=f'{selected} 기준', line=dict(color='#0f766e')))
                overlay.add_trace(go.Scatter(x=np.linspace(0,100,n), y=100*np.exp(resample(np.log(candidate.Close/candidate.Close.iloc[0]),n)), name=r['종목'], line=dict(color='#ea580c')))
                overlay.update_layout(title='가격 패턴 (시작=100)',xaxis_title='구간 진행률 (%)',height=360)
                st.plotly_chart(overlay, use_container_width=True)
            with c2:
                vol = go.Figure()
                for data,name,color in [(ref,'기준','#0f766e'),(candidate,r['종목'],'#ea580c')]:
                    logvol=np.log1p(data.Volume.to_numpy(dtype=float))
                    center=pd.Series(logvol).rolling(20,min_periods=1).median().to_numpy()
                    vol.add_trace(go.Scatter(x=np.linspace(0,100,n),y=resample(np.clip(logvol-center,-4,6),n),name=name,line=dict(color=color)))
                vol.update_layout(title='상대 거래량 패턴',xaxis_title='구간 진행률 (%)',height=360)
                st.plotly_chart(vol,use_container_width=True)
            st.caption(f"검색 구간: {r['시작'].date()} ~ {r['종료'].date()} · 종합 {r['유사도']}점. 점수는 유사도 지표이며 상승 확률이 아닙니다.")

st.divider()
st.caption('데이터: FinanceDataReader / pykrx · 분석: 제약 DTW + 추세 통계 · 현재 상장 종목만 포함(생존 편향). 금융투자 조언이 아닙니다.')
