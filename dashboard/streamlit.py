import streamlit as st
import pandas as pd
import sqlalchemy as sa
import requests
from datetime import datetime
import plotly.express as px
import numpy as np
import os
import joblib


# -----------------------------------------------------------------
# 1. DB 연결 설정
# -----------------------------------------------------------------
def get_db_connection():
    """SQLAlchemy 연결 엔진을 생성합니다."""
    try:
        db_info = st.secrets["database"]
        engine_url = (
            f"mysql+mysqlconnector://{db_info['user']}:{db_info['password']}@"
            f"{db_info['host']}:{db_info['port']}/{db_info['db_name']}"
        )
        engine = sa.create_engine(engine_url, pool_pre_ping=True)
        return engine
    except Exception:
        st.info("DB 비사용 모드: .streamlit/secrets.toml의 [database] 설정이 없거나 연결 실패")
        return None


@st.cache_resource
def init_db_connection():
    """DB 연결 엔진을 초기화하고 반환합니다."""
    return get_db_connection()


# -----------------------------------------------------------------
# 2. 데이터 로드 함수들
# -----------------------------------------------------------------
@st.cache_data(ttl=600)
def load_data_from_db(_engine, table_name, limit=1000, order_by_col='timestamp'):
    """DB에서 데이터를 로드하는 범용 함수 (★수정★: 델타 로직 제거, 원래대로 복원)"""
    if _engine is None:
        return pd.DataFrame()
    try:
        order_clause = f"ORDER BY {order_by_col} DESC" if order_by_col else ""

        # 'None' 문자열이 아닌 진짜 None 타입으로 limit 처리
        if limit == 'None' or limit is None:
            limit_clause = ""
        else:
            limit_clause = f"LIMIT {limit}"

        query = f"SELECT * FROM {table_name} {order_clause} {limit_clause}"

        df = pd.read_sql(query, con=_engine)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'fcst_dt' in df.columns:  # 날씨 테이블용
            df['fcst_dt'] = pd.to_datetime(df['fcst_dt'])
        elif 'fcst_date' in df.columns:
            df['fcst_date'] = pd.to_datetime(df['fcst_date'])
        return df
    except Exception as e:
        st.error(f"'{table_name}' 데이터 로드 오류: {e}")
        return pd.DataFrame()

# -----------------------------------------------------------------
# 3. AI 모델 로드 함수
# -----------------------------------------------------------------
@st.cache_resource
def load_prediction_model(model_path):
    """(방법 1) 학습된 AI 모델 파일을 로드합니다."""
    if not os.path.exists(model_path):
        st.warning(f"모델 파일({model_path})을 찾을 수 없습니다. AI Mock-up 모드로 작동합니다.")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        return None


# -----------------------------------------------------------------
# 4. 페이지 이동(드릴다운)을 위한 함수
# -----------------------------------------------------------------
def set_detail_view(chamber_id, chamber_no):
    st.session_state.view_mode = 'detail'
    st.session_state.selected_chamber_id = chamber_id
    st.session_state.selected_chamber_no = chamber_no


def set_overview_view():
    st.session_state.view_mode = 'overview'
    st.session_state.selected_chamber_id = None
    st.session_state.selected_chamber_no = None


# -----------------------------------------------------------------
# 5. Streamlit 대시보드 UI 구성
# -----------------------------------------------------------------

st.set_page_config(page_title="🐷 스마트 축사 대시보드", layout="wide")

if 'view_mode' not in st.session_state:
    set_overview_view()

# --- 1. 모든 원본 데이터 로드 ---
engine = init_db_connection()
# limit 원래대로 복원
sensor_df_all = load_data_from_db(engine, 'Chamber_Logs', limit=20000)
equipment_df_all = load_data_from_db(engine, 'Equipment_Logs', limit=20000)
weather_ultra_fcst_df = load_data_from_db(engine, "weather_ultra_fcst", limit=48, order_by_col="fcst_dt")

pig_log_df_all = load_data_from_db(engine, 'Pig_Logs', limit='None', order_by_col='timestamp')
chambers_df = load_data_from_db(engine, 'Chambers', limit='None', order_by_col=None)
pigs_df = load_data_from_db(engine, 'Pigs', limit='None', order_by_col=None)

mid_land_fcst_df = load_data_from_db(
    engine,
    "mid_land_fcst",
    limit=7,
    order_by_col="fcst_date"
)
weather_ultra_fcst_df = load_data_from_db(
    engine,
    "weather_ultra_fcst",
    limit=48,
    order_by_col="fcst_dt"
)

if 'weight_kg' in pig_log_df_all.columns:
    pig_log_df_all['weight_kg'] = pd.to_numeric(pig_log_df_all['weight_kg'], errors='coerce')

if not weather_ultra_fcst_df.empty:
    weather_ultra_fcst_df.columns = [col.upper() for col in weather_ultra_fcst_df.columns]

# =================================================================
# A. '전체 맵 (Overview)' 화면
# =================================================================
if st.session_state.view_mode == "overview":

    st.title("🐷 스마트 축사 현황 (전체 맵)")

    # --- 1. 새로운 '정상' 건강 기준 정의 ---
    temp_norm_min = 37.0
    temp_norm_max = 39.0
    breath_norm_min = 55
    breath_norm_max = 65

    with st.container(border=True):
        st.subheader("AICU 총괄 요약")
        cols = st.columns(5)

        if not pig_log_df_all.empty:
            total_pigs = len(pig_log_df_all['pig_id'].unique())
            cols[0].metric("총 사육 두수", f"{total_pigs} 마리")
        else:
            cols[0].metric("총 사육 두수", "N/A (로그 없음)")

        # 3. '총 주의 개체 수' (새로운 기준으로 계산)
        if not pig_log_df_all.empty:
            # (데이터 타입 변환 및 유효 데이터 필터링)
            try:
                pig_log_df_all['temp_rectal'] = pd.to_numeric(pig_log_df_all['temp_rectal'], errors='coerce')
                pig_log_df_all['breath_rate'] = pd.to_numeric(pig_log_df_all['breath_rate'], errors='coerce')
            except Exception:
                pass  # 오류 무시

            valid_logs = pig_log_df_all.dropna(subset=['temp_rectal', 'breath_rate'])

            if not valid_logs.empty:
                latest_pig_logs = valid_logs.loc[valid_logs.groupby("pig_id")["timestamp"].idxmax()]

                # '정상' 범위를 벗어나는 모든 개체 필터링
                warning_pigs_total = latest_pig_logs[
                    (latest_pig_logs["temp_rectal"] < temp_norm_min) |
                    (latest_pig_logs["temp_rectal"] > temp_norm_max) |
                    (latest_pig_logs["breath_rate"] < breath_norm_min) |
                    (latest_pig_logs["breath_rate"] > breath_norm_max)
                    ]
                cols[1].metric("총 '주의' 개체 수", f"{len(warning_pigs_total)} 마리")
            else:
                cols[1].metric("총 '주의' 개체 수", "N/A (데이터 부족)")
        else:
            cols[1].metric("총 '주의' 개체 수", "N/A (로그 없음)")

        # 3. '현재 외부 날씨' (시간별 DB 데이터 사용)
        if not weather_ultra_fcst_df.empty and {"T1H", "REH"}.issubset(weather_ultra_fcst_df.columns):
            latest_weather = weather_ultra_fcst_df.iloc[0]  # 가장 최신 시간
            cols[2].metric("현재 외부 온도", f"{latest_weather.get('T1H', 0):.1f} °C")
            cols[3].metric("현재 외부 습도", f"{latest_weather.get('REH', 0):.1f} %")
        else:
            cols[2].metric("현재 외부 온도", "N/A")
            cols[3].metric("현재 외부 습도", "N/A")

        # 4. '오늘 강수 확률' (일일 요약 DB 데이터 사용)
        if not mid_land_fcst_df.empty and {"pop_am", "pop_pm"}.issubset(mid_land_fcst_df.columns):
            today_weather = mid_land_fcst_df.iloc[0]  # 오늘 예보
            pop_am = today_weather.get("pop_am", 0)  # 오전 강수 확률
            pop_pm = today_weather.get("pop_pm", 0)  # 오후 강수 확률
            cols[4].metric("오전/오후 강수", f"{pop_am}% / {pop_pm}%")
            if (pop_am or 0) > 70 or (pop_pm or 0) > 70:
                st.warning("🚨 강수 확률 70% 이상! 환기/습도 관리에 유의하세요.")
        else:
            cols[4].metric("강수 확률", "N/A")

    # ----------------------------------------------------

    st.divider()
    st.subheader("챔버별 현황 (클릭하여 드릴다운)")

    if chambers_df.empty:
        st.error("챔버 정보를 찾을 수 없습니다.")
    else:
        grid_cols = st.columns(2)

        for i, row in chambers_df.iterrows():
            chamber_id = row['chamber_id']
            chamber_no = row['chamber_no']
            current_col = grid_cols[i % 2]

            # 5. 챔버별 '주의' 개체 수 (새로운 기준으로 계산)
            warn_count = 0
            if not pigs_df.empty and not pig_log_df_all.empty:
                pigs_in_chamber_ids = pigs_df[pigs_df['chamber_id'] == chamber_id]['pig_id']
                pig_logs_in_chamber = pig_log_df_all[pig_log_df_all['pig_id'].isin(pigs_in_chamber_ids)]

                # (유효한 건강 데이터만 필터링)
                valid_logs_in_chamber = pig_logs_in_chamber.dropna(subset=['temp_rectal', 'breath_rate'])

                if not valid_logs_in_chamber.empty:
                    latest_pig_logs_chamber = valid_logs_in_chamber.loc[
                        valid_logs_in_chamber.groupby('pig_id')['timestamp'].idxmax()]

                    # '정상' 범위를 벗어나는 개체 필터링
                    warning_pigs_chamber = latest_pig_logs_chamber[
                        (latest_pig_logs_chamber["temp_rectal"] < temp_norm_min) |
                        (latest_pig_logs_chamber["temp_rectal"] > temp_norm_max) |
                        (latest_pig_logs_chamber["breath_rate"] < breath_norm_min) |
                        (latest_pig_logs_chamber["breath_rate"] > breath_norm_max)
                        ]
                    warn_count = len(warning_pigs_chamber)

            # 6. '주의' 개체 수(warn_count)에 따라 컨테이너 제목 변경
            with current_col.container(border=True):
                if warn_count > 0:
                    st.error(f"🚨 {chamber_no}번 챔버 (주의!)")  # (주의 개체가 1명이라도 있으면 에러 표시)
                else:
                    st.subheader(f"✅ {chamber_no}번 챔버")

                c1_metric, c2_metric = st.columns(2)

                chamber_sensor_data = sensor_df_all[sensor_df_all['chamber_id'] == chamber_id]
                if not chamber_sensor_data.empty and "temperature" in chamber_sensor_data.columns:
                    # .iloc[0] 추가
                    c1_metric.metric("현재 온도", f"{chamber_sensor_data.iloc[0]['temperature']:.1f} °C")
                else:
                    c1_metric.metric("현재 온도", "N/A")

                # 7. 계산된 'warn_count'를 정확히 표시
                c2_metric.metric("건강 '주의' 개체", f"{warn_count} 마리")

                st.button(
                    f"{chamber_no}번 챔버 상세 정보 보기",
                    key=f"btn_detail_{chamber_id}",
                    on_click=set_detail_view,
                    args=(chamber_id, chamber_no)
                )
    # ('주간 날씨 예보' 테이블)
    # ----------------------------------------------------
    st.divider()
    st.subheader("🗓️ 주간 날씨 요약 (기상청 DB)")

    # (DB에서 로드한 mid_land_fcst_df 변수 사용)
    needed_cols = ["fcst_date", "wf_am", "pop_am", "wf_pm", "pop_pm", "tmin", "tmax"]

    if not mid_land_fcst_df.empty and all(col in mid_land_fcst_df.columns for col in needed_cols):

        # 1. 대시보드에 표시할 컬럼만 선택
        display_df = mid_land_fcst_df[list(needed_cols)].copy()

        # 2. 날짜순으로 정렬
        display_df = display_df.sort_values(by="fcst_date")

        # 3. 날짜 형식을 '00월 00일 (요일)'로 변경
        display_df['fcst_date'] = display_df['fcst_date'].dt.strftime('%m월 %d일 (%a)')

        # 4. 컬럼 이름을 한글로 변경
        display_df = display_df.rename(columns={
            "fcst_date": "날짜",
            "pop_am": "오전 확률(%)",
            "wf_am": "오전 날씨",
            "pop_pm": "오후 확률(%)",
            "wf_pm": "오후 날씨",
            "tmin": "최저 기온(°C)",
            "tmax": "최고 기온(°C)"
        })

        display_df['일일 강수 확률(%)'] = display_df[['오전 확률(%)', '오후 확률(%)']].max(axis=1).astype(int)

        weather_emoji_map = {
            "맑음": "☀️",
            "구름많음": "🌥️",
            "흐림": "☁️",
            "비": "🌧️",
            "눈": "❄️",
            "비/눈": "🌨️",
            "소나기": "🌦️"
            # (필요시 DB에 있는 다른 텍스트도 추가)
        }
        # 2. '오전 날씨'와 '오후 날씨' 컬럼의 텍스트를 이모티콘으로 바꿉니다.
        display_df["오전 날씨"] = display_df["오전 날씨"].replace(weather_emoji_map)
        display_df["오후 날씨"] = display_df["오후 날씨"].replace(weather_emoji_map)

        final_column_order = [
            "날짜",
            "일일 강수 확률(%)",
            "오전 날씨",
            "오후 날씨",
            "최저 기온(°C)",
            "최고 기온(°C)",
        ]
        display_df = display_df[final_column_order]
        # 5. '날짜'를 인덱스로 설정하여 테이블(표)로 표시
        st.dataframe(
            display_df.set_index("날짜"),
            width='stretch'  # (use_container_width=True 대신 사용)
        )

    else:
        st.warning("주간 날씨 요약(mid_land_fcst) 데이터를 DB에서 불러오지 못했거나, 필요한 컬럼이 없습니다.")

# =================================================================
# B. '챔버 상세 (Detail)' 화면
# =================================================================
elif st.session_state.view_mode == 'detail':

    st.button("◀ 전체 맵으로 돌아가기", on_click=set_overview_view)
    selected_id = st.session_state.selected_chamber_id
    selected_no = st.session_state.selected_chamber_no
    st.title(f"🐷 {selected_no}번 챔버 상세 정보")

    sensor_df_filtered = sensor_df_all[sensor_df_all['chamber_id'] == selected_id]
    equipment_df_filtered = equipment_df_all[equipment_df_all['chamber_id'] == selected_id]

    pig_log_df_filtered = pd.DataFrame()
    if not pigs_df.empty:
        pigs_in_chamber = pigs_df[pigs_df['chamber_id'] == selected_id]['pig_id']
        pig_log_df_filtered = pig_log_df_all[pig_log_df_all['pig_id'].isin(pigs_in_chamber)]

    st.divider()

    st.header("📈 현재 챔버 상황")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 환경 센서 (Chamber_Logs)")
        if not sensor_df_filtered.empty:
            latest_sensor = sensor_df_filtered.iloc[0]
            c1, c2, c3 = st.columns(3)
            #가짜 델타 제거
            c1.metric("온도", f"{latest_sensor['temperature']:.1f} °C")
            c2.metric("습도", f"{latest_sensor['humidity']:.1f} %")
            c3.metric("CO2", f"{latest_sensor['co2']:.0f} ppm")

            min_date = sensor_df_filtered['timestamp'].min().date()
            max_date = sensor_df_filtered['timestamp'].max().date()

            date_range = st.date_input(
                "조회 기간을 선택하세요:",
                value=(min_date, max_date), min_value=min_date, max_value=max_date,
                key=f"date_selector_{selected_id}"
            )

            chart_data_filtered_by_date = pd.DataFrame()
            if len(date_range) == 2:
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                chart_data_filtered_by_date = sensor_df_filtered[
                    (sensor_df_filtered['timestamp'] >= start_date) &
                    (sensor_df_filtered['timestamp'] <= end_date)
                    ]

            if chart_data_filtered_by_date.empty:
                st.info("선택된 기간에 해당하는 센서 데이터가 없습니다.")
            else:
                tab1_chart, tab2_chart, tab3_chart = st.tabs(["🌡️ 온도", "💧 습도", "💨 CO2"])
                with tab1_chart:
                    fig_temp = px.line(chart_data_filtered_by_date, x='timestamp', y='temperature', title='온도 추이')
                    st.plotly_chart(fig_temp, width='stretch')
                with tab2_chart:
                    fig_humi = px.line(chart_data_filtered_by_date, x='timestamp', y='humidity', title='습도 추이')
                    st.plotly_chart(fig_humi, width='stretch')
                with tab3_chart:
                    fig_co2 = px.line(chart_data_filtered_by_date, x='timestamp', y='co2', title='CO2 추이')
                    st.plotly_chart(fig_co2, width='stretch')

        else:
            st.warning("센서 데이터를 찾을 수 없습니다.")

    with col2:
        st.subheader("❤️ 돼지 건강 상태 (Pig_Logs)")

        # (Pig_Logs 데이터가 필터링되어 'pig_log_df_filtered'에 있다고 가정)
        if not pig_log_df_filtered.empty:

            # 1. 새로운 '정상' 범위 정의
            temp_norm_min = 37.0
            temp_norm_max = 39.0
            breath_norm_min = 55
            breath_norm_max = 65

            # 2. 데이터 타입 변환 및 유효 데이터 필터링
            # (weight_kg와 마찬가지로, 숫자 변환 및 NaN/NULL 값 제거)
            try:
                pig_log_df_filtered['temp_rectal'] = pd.to_numeric(pig_log_df_filtered['temp_rectal'], errors='coerce')
                pig_log_df_filtered['breath_rate'] = pd.to_numeric(pig_log_df_filtered['breath_rate'], errors='coerce')
            except Exception as e:
                st.warning(f"건강 상태 분석 중 타입 변환 오류: {e}")

            valid_health_logs = pig_log_df_filtered.dropna(subset=['temp_rectal', 'breath_rate'])

            if not valid_health_logs.empty:
                # 3. 각 돼지의 가장 최신 로그 가져오기
                latest_pig_logs = valid_health_logs.loc[valid_health_logs.groupby('pig_id')['timestamp'].idxmax()]

                # 4. '정상' 범위를 벗어나는 모든 개체 필터링
                warning_pigs = latest_pig_logs[
                    (latest_pig_logs['temp_rectal'] < temp_norm_min) |  # 온도 낮음
                    (latest_pig_logs['temp_rectal'] > temp_norm_max) |  # 온도 높음
                    (latest_pig_logs['breath_rate'] < breath_norm_min) |  # 호흡 느림
                    (latest_pig_logs['breath_rate'] > breath_norm_max)  # 호흡 빠름
                    ]

                st.metric("건강 '주의' 개체 수", f"{len(warning_pigs)} 마리")

                if len(warning_pigs) > 0:
                    with st.expander("'주의' 개체 목록 보기"):

                        # 5. '주의 원인'을 찾는 함수 로직 변경
                        def find_reason(row):
                            reasons = []
                            # 온도 확인
                            if row['temp_rectal'] < temp_norm_min:
                                reasons.append(f"온도 낮음 ({row['temp_rectal']:.1f}°C)")
                            elif row['temp_rectal'] > temp_norm_max:
                                reasons.append(f"온도 높음 ({row['temp_rectal']:.1f}°C)")

                            # 호흡 확인
                            if row['breath_rate'] < breath_norm_min:
                                reasons.append(f"호흡 느림 ({row['breath_rate']:.0f}회)")
                            elif row['breath_rate'] > breath_norm_max:
                                reasons.append(f"호흡 빠름 ({row['breath_rate']:.0f}회)")

                            return ', '.join(reasons)


                        warning_pigs_with_reason = warning_pigs.copy()
                        warning_pigs_with_reason['주의 원인'] = warning_pigs_with_reason.apply(find_reason, axis=1)

                        # 데이터프레임에 표시할 컬럼 (순서 지정)
                        display_cols = ["pig_id", "temp_rectal", "breath_rate", "주의 원인"]
                        st.dataframe(warning_pigs_with_reason[display_cols])
            else:
                st.warning("유효한 건강 데이터(체온/호흡수)가 없습니다.")
        else:
            st.warning("돼지 로그 데이터를 찾을 수 없습니다.")

    st.divider()

    # 챔버 외부 날씨 (시간별 상세 예보 DB)
    st.header("🌦️ 챔버 외부 날씨 (기상청 DB)")

    # (대문자로 변환된 컬럼명 사용)
    needed_weather_cols = {"FCST_DT", "T1H", "REH", "RN1", "SKY", "PTY"}

    if not weather_ultra_fcst_df.empty and needed_weather_cols.issubset(weather_ultra_fcst_df.columns):

        weather_chart_data = weather_ultra_fcst_df.set_index("FCST_DT")

        w_tab1, w_tab2, w_tab3 = st.tabs(["🌡️ 외부 기온 (T1H)", "💧 외부 습도 (REH)", "☔ 시간당 강수량 (RN1)"])

        with w_tab1:
            st.plotly_chart(px.line(weather_chart_data, y='T1H', title='시간별 외부 기온'), width='stretch')
        with w_tab2:
            st.plotly_chart(px.line(weather_chart_data, y='REH', title='시간별 외부 습도'), width='stretch')
        with w_tab3:
            st.plotly_chart(px.bar(weather_chart_data, y='RN1', title='시간별 강수량'), width='stretch')

        latest_sky = weather_ultra_fcst_df.iloc[0].get("SKY", -1)
        st.info(f"참고: 현재 하늘 상태(SKY) 코드는 '{latest_sky}'입니다. (1: 맑음, 3: 구름많음, 4: 흐림)")

    else:
        st.warning("시간별 상세 날씨(weather_ultra_fcst) 데이터를 DB에서 불러오지 못했거나, 필요한 컬럼이 없습니다.")

    st.divider()

    # --- 섹션 3: 출하 및 에너지 분석 ---
    st.header("🐖 출하 및 에너지 분석")
    tab1, tab2 = st.tabs(["출하 날짜 예측", "에너지 사용량 분석"])

    with tab1:
        target_weight = st.number_input(
            "목표 출하 체중(kg)을 입력하세요:",
            min_value=80.0, value=80.0, step=1.0,
            help="이 체중을 기준으로 출하 가능 개체 수와 예측 날짜를 계산합니다."
        )

        if not pig_log_df_filtered.empty:
            valid_latest_weights = pig_log_df_filtered.loc[pig_log_df_filtered.groupby('pig_id')['timestamp'].idxmax()]
            valid_latest_weights = valid_latest_weights.dropna(subset=['weight_kg'])

            if valid_latest_weights.empty:
                st.warning("이 챔버에는 유효한 체중 기록이 없습니다.")
            else:
                ship_ready_now = valid_latest_weights[valid_latest_weights['weight_kg'] >= target_weight]

                col1_ship, col2_ship = st.columns(2)
                col1_ship.metric(f"현재 {target_weight}kg 이상 (출하 가능)", f"{len(ship_ready_now)} 마리")
                col2_ship.metric("1주일 내 출하 가능 (Mock)", f"{int(len(ship_ready_now) * 0.5) + 2} 마리 (Mock)")
                st.divider()

                st.subheader(f"🐷 {target_weight}kg 도달 날짜 예측 (AI Mock-up)")
                pigs_below_target = valid_latest_weights[valid_latest_weights['weight_kg'] < target_weight]

                if not pigs_below_target.empty:
                    prediction_df = pigs_below_target.copy()
                    ADG = 0.7
                    today = pd.Timestamp.now()

                    prediction_df['부족한 체중(kg)'] = target_weight - prediction_df['weight_kg']
                    prediction_df['예상 소요 일수'] = prediction_df['부족한 체중(kg)'] / ADG
                    prediction_df['예상 출하일'] = prediction_df['예상 소요 일수'].apply(
                        lambda days: today + pd.Timedelta(days=days))
                    prediction_df = prediction_df.sort_values('예상 출하일', ascending=True)

                    display_cols = ['pig_id', 'weight_kg', '예상 소요 일수', '예상 출하일']
                    prediction_df_display = prediction_df[display_cols].rename(columns={
                        'pig_id': '돼지 ID', 'weight_kg': '현재 체중(kg)',
                        '예상 소요 일수': '남은 일수(일)', '예상 출하일': '예상 출하 날짜'
                    })

                    prediction_df_display['남은 일수(일)'] = prediction_df_display['남은 일수(일)'].round(0).astype(int)
                    prediction_df_display['예상 출하 날짜'] = prediction_df_display['예상 출하 날짜'].dt.strftime('%Y-%m-%d')
                    prediction_df_display['현재 체중(kg)'] = prediction_df_display['현재 체중(kg)'].round(1)

                    fastest_pig = prediction_df_display.iloc[0]
                    st.metric(
                        f"가장 빠른 예상 출하일 (ID: {fastest_pig['돼지 ID']})",
                        f"{fastest_pig['예상 출하 날짜']}",
                        f"{fastest_pig['남은 일수(일)']}일 남음"
                    )

                    with st.expander("전체 개체별 예상 출하일 보기 (빠른 순)"):
                        st.dataframe(prediction_df_display, width='stretch')

                else:
                    if not ship_ready_now.empty:
                        st.success(f"모든 개체가 이미 목표 체중({target_weight}kg) 이상입니다.")
                    else:
                        st.info("분석할 유효한 체중 데이터가 없습니다 (모두 NaN일 수 있음).")
        else:
            st.warning("몸무게 데이터가 없어 계산할 수 없습니다.")

    with tab2:
        if not equipment_df_filtered.empty:
            min_date_eq = equipment_df_filtered['timestamp'].min().date()
            max_date_eq = equipment_df_filtered['timestamp'].max().date()

            date_range_eq = st.date_input(
                "조회 기간을 선택하세요:",
                value=(min_date_eq, max_date_eq),
                min_value=min_date_eq,
                max_value=max_date_eq,
                key=f"energy_date_selector_{selected_id}"
            )

            energy_data_filtered_by_date = pd.DataFrame()
            start_date_str = min_date_eq.isoformat()
            end_date_str = max_date_eq.isoformat()

            if len(date_range_eq) == 2:
                start_date = pd.to_datetime(date_range_eq[0])
                end_date = pd.to_datetime(date_range_eq[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                start_date_str = date_range_eq[0].isoformat()
                end_date_str = date_range_eq[1].isoformat()

                energy_data_filtered_by_date = equipment_df_filtered[
                    (equipment_df_filtered['timestamp'] >= start_date) &
                    (equipment_df_filtered['timestamp'] <= end_date)
                    ]

            if energy_data_filtered_by_date.empty:
                st.info("선택된 기간에 해당하는 에너지 데이터가 없습니다.")
            else:
                st.subheader(f"기간 내 장비별 사용량 ({start_date_str} ~ {end_date_str})")
                period_usage = energy_data_filtered_by_date.groupby('equipment_type')['power_usage_wh'].sum() / 1000
                fig_energy_period = px.bar(period_usage, title="장비별 기간 내 사용량 (kWh)",
                                           labels={'value': '사용량 (kWh)', 'equipment_type': '장비 종류'})
                st.plotly_chart(fig_energy_period, width='stretch')

                st.divider()


                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False, encoding='utf-8-sig')


                csv_data = convert_df_to_csv(energy_data_filtered_by_date)

                st.download_button(
                    label=f"📈 기간({start_date_str}~{end_date_str}) 로그 다운로드",
                    data=csv_data,
                    file_name=f"energy_logs_{selected_no}ch_{start_date_str}_to_{end_date_str}.csv",
                    mime="text/csv",
                )
        else:
            st.warning("에너지 사용량 데이터가 없습니다.")

    st.divider()

    # --- 섹션 4: AI 예측 결과 (XAI 기능 포함) ---
    st.header("🤖 AI 예측 결과")

    MODEL_FILE_PATH = "shipment_model.pkl"
    model = load_prediction_model(MODEL_FILE_PATH)

    if model is None:
        st.info("AI 모델 파일(shipment_model.pkl)을 찾을 수 없습니다. Mock-up 모드로 UI를 표시합니다.")
        st.subheader("🐖 계절별 출하 분류 (테스트)")
        col1, col2 = st.columns(2)
        col1.metric("예측 결과", "정상 출하");
        col2.metric("정상 확률", "90 %")

        st.subheader("AI 판단 근거 (XAI Mock-up)")
        shap_values = pd.DataFrame({
            '영향력': [0.12, 0.05, -0.08],
            '색상': ['blue', 'blue', 'red']
        }, index=['온도(긍정)', '습도(긍정)', 'CO2(부정)'])
        st.bar_chart(shap_values, y='영향력', color='색상')
        st.info("파란색 막대는 '정상' 예측에 긍정적인 영향을, 빨간색 막대는 부정적인 영향을 준 요인입니다.")

    elif not sensor_df_filtered.empty:
        try:
            latest_data = sensor_df_filtered.sort_values("timestamp").tail(1).iloc[0]

            features_df = pd.DataFrame({
                'temperature': [latest_data['temperature']],
                'humidity': [latest_data['humidity']],
                'co2': [latest_data['co2']]
                # ... (모델 학습에 사용한 다른 모든 컬럼 추가)
            })

            prediction = model.predict(features_df)
            prediction_proba = model.predict_proba(features_df)

            st.subheader("🐖 계절별 출하 분류 (AI 예측)")
            col1, col2 = st.columns(2)
            col1.metric("예측 결과", f"{prediction[0]}")
            col2.metric("정상 확률", f"{prediction_proba[0][1] * 100:.0f} %")

            st.subheader("AI 판단 근거 (XAI)")
            st.info("실제 SHAP 라이브러리를 연동하여 AI가 왜 이런 예측을 했는지 시각화할 수 있습니다.")
            with st.expander("AI 예측에 사용된 입력값 보기"):
                st.dataframe(features_df)

        except KeyError as e:
            st.warning(f"AI 예측에 필요한 컬럼({e})이 DB 데이터에 없습니다.")
        except Exception as e:
            st.error(f"AI 예측 중 오류 발생: {e}")
    else:
        st.info("센서 데이터가 없어 AI 예측 UI를 표시할 수 없습니다.")
