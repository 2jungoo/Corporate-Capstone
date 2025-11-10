import os, datetime as dt, pandas as pd, numpy as np
import streamlit as st, sqlalchemy as sa

st.set_page_config(page_title="날씨 대시보드", layout="wide")
DB=os.environ.get("DB_URL","")
engine=sa.create_engine(DB, pool_pre_ping=True) if DB else None

@st.cache_data(ttl=120)
def q_mid(reg, s, e):
    sql = sa.text("""
        SELECT reg_id, fcst_date, wf_am, wf_pm, pop_am, pop_pm, tmin, tmax, tmfc
        FROM mid_land_fcst
        WHERE reg_id=:r AND fcst_date BETWEEN :s AND :e
        ORDER BY fcst_date
    """)
    return pd.read_sql(sql, engine, params={"r": reg, "s": s, "e": e})

@st.cache_data(ttl=120)
def q_ultra(nx, ny, s, e):
    sql = sa.text("""
        SELECT fcst_dt, T1H, REH, RN1, WSD, SKY, PTY
        FROM weather_ultra_fcst
        WHERE nx=:nx AND ny=:ny AND fcst_dt>=:s AND fcst_dt<:e
        ORDER BY fcst_dt
    """)
    return pd.read_sql(sql, engine, params={"nx": nx, "ny": ny, "s": s, "e": e})

def map_weather_label(sky, pty):
    if pty in [1,2,5,6]: return "비"
    if pty in [3,7]: return "눈"
    if pty==4: return "소나기"
    if sky==1: return "맑음"
    if sky==3: return "구름많음"
    if sky==4: return "흐림"
    return "맑음"

st.title("중기·초단기 날씨 대시보드")

with st.sidebar:
    st.header("필터")
    reg = st.text_input("reg_id", value="11B00000")
    nx = st.number_input("nx", value=60, step=1)
    ny = st.number_input("ny", value=127, step=1)
    today = dt.date.today()
    mid_range = st.date_input("일간 기간", (today, today + dt.timedelta(days=7)))
    ultra_day = st.date_input("시간별 날짜", today)
    refresh = st.button("새로고침")

tab1, tab2 = st.tabs(["일간 요약", "시간별 상세"])

with tab1:
    if engine is None:
        st.error("DB_URL이 설정되어 있지 않습니다.")
    else:
        if isinstance(mid_range, tuple) and len(mid_range)==2:
            s, e = mid_range
        else:
            s, e = today, today + dt.timedelta(days=7)
        df = q_mid(reg, s.isoformat(), e.isoformat()) if not refresh else q_mid.clear() or q_mid(reg, s.isoformat(), e.isoformat())
        if df.empty:
            st.warning("일간 데이터 없음")
        else:
            c1, c2 = st.columns([2,1])
            with c1:
                st.subheader("일자별 요약")
                st.dataframe(df, use_container_width=True)
            with c2:
                sel = st.selectbox("요약 보기", df["fcst_date"].tolist())
                sub = df[df["fcst_date"]==sel].iloc[0]
                st.metric("최저기온(°C)", sub["tmin"])
                st.metric("최고기온(°C)", sub["tmax"])
                st.metric("오전 강수확률(%)", sub["pop_am"])
                st.metric("오후 강수확률(%)", sub["pop_pm"])
                st.write(f"오전: {sub['wf_am']}")
                st.write(f"오후: {sub['wf_pm']}")
                st.caption(f"기준: {sub['tmfc']}")
            g = df[["fcst_date","tmin","tmax"]].copy()
            g = g.set_index("fcst_date")
            st.subheader("최저/최고 기온")
            st.line_chart(g)
            p = df[["fcst_date","pop_am","pop_pm"]].set_index("fcst_date")
            st.subheader("강수확률")
            st.bar_chart(p)

with tab2:
    if engine is None:
        st.error("DB_URL이 설정되어 있지 않습니다.")
    else:
        s = dt.datetime.combine(ultra_day, dt.time(0,0,0))
        e = s + dt.timedelta(days=1)
        dfh = q_ultra(nx, ny, s, e) if not refresh else q_ultra.clear() or q_ultra(nx, ny, s, e)
        if dfh.empty:
            st.warning("시간별 데이터 없음")
        else:
            dfh["label"] = [map_weather_label(int(sky) if pd.notna(sky) else 1, int(pty) if pd.notna(pty) else 0) for sky,pty in zip(dfh["SKY"], dfh["PTY"])]
            c1, c2, c3 = st.columns(3)
            now_row = dfh.iloc[0]
            c1.metric("평균기온(°C)", round(dfh["T1H"].mean(),1))
            c2.metric("최대강수량(mm)", float(dfh["RN1"].fillna(0).max()))
            c3.metric("평균습도(%)", int(round(dfh["REH"].mean() if pd.notna(dfh['REH']).any() else 0)))
            st.subheader("시간별 기온(°C)")
            st.line_chart(dfh.set_index("fcst_dt")[["T1H"]])
            st.subheader("시간별 습도(%)")
            if "REH" in dfh.columns:
                st.line_chart(dfh.set_index("fcst_dt")[["REH"]])
            st.subheader("시간별 강수량(mm)")
            st.bar_chart(dfh.set_index("fcst_dt")[["RN1"]].fillna(0))
            st.subheader("시간별 표")
            show = dfh.copy()
            show["fcst_dt"] = pd.to_datetime(show["fcst_dt"]).dt.strftime("%m-%d %H:%M")
            st.dataframe(show, use_container_width=True)
