# === Smart Farm Dashboard (Merged: latest UI + auth/FAB + weekly emojis + hybrid AI) ===
import os
import warnings
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import sqlalchemy as sa
import streamlit as st

# Optional heavy deps for hybrid predictor
try:
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping
except Exception:
    keras = None
    layers = None
    EarlyStopping = None

try:
    import xgboost as xgb  # noqa: F401
except Exception:
    xgb = None

# -----------------------------
# Optional auth integration
# -----------------------------
try:
    from auth import current_user, logout, require_perms  # type: ignore
except Exception:
    current_user = lambda: {"id": "anonymous"}  # noqa: E731


    def logout():
        pass


    def require_perms(_):
        pass


# -----------------------------
# Page config + FAB logout
# -----------------------------


def render():
    st.set_page_config(page_title="🐷 스마트 축사 대시보드", layout="wide")

    # Handle ?logout=1
    try:
        try:
            q = st.query_params
        except AttributeError:
            q = st.experimental_get_query_params()
        v = q.get("logout", "")
        if isinstance(v, list):
            v = v[0] if v else ""
        if str(v) == "1":
            logout()
            try:
                st.query_params.clear()
            except Exception:
                st.experimental_set_query_params()
            st.rerun()
    except Exception:
        pass

    # Gate by login/permission if available
    try:
        u = current_user()
        if not u:
            try:
                st.switch_page("login.py")
            except Exception:
                st.error("로그인이 필요합니다.")
                st.stop()
        else:
            try:
                require_perms(["view_dashboard"])
            except Exception:
                pass
    except Exception:
        pass

    # Hide login/Signup links if using multipage
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] a[href$="/login"]{display:none !important;}
    section[data-testid="stSidebar"] a[href$="/Signup"]{display:none !important;}
    a.fab-logout{
      position: fixed; right: 22px; bottom: 22px;
      background: #ff4d4f; color: #fff; padding: 12px 18px;
      border-radius: 999px; text-decoration: none; font-weight: 700;
      box-shadow: 0 8px 18px rgba(0,0,0,.25); z-index: 9999;
    }
    a.fab-logout:hover{ filter: brightness(0.95); }
    </style>
    """, unsafe_allow_html=True)

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
    # 2. 데이터 로드
    # -----------------------------------------------------------------
    @st.cache_data(ttl=600)
    def load_data_from_db(_engine, table_name, limit=1000, order_by_col='timestamp'):
        """DB에서 데이터를 로드하는 범용 함수"""
        if _engine is None:
            return pd.DataFrame()
        try:
            order_clause = f"ORDER BY {order_by_col} DESC" if order_by_col else ""
            if limit == 'None' or limit is None:
                limit_clause = ""
            else:
                limit_clause = f"LIMIT {limit}"
            query = f"SELECT * FROM {table_name} {order_clause} {limit_clause}"
            df = pd.read_sql(query, con=_engine)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'fcst_dt' in df.columns:
                df['fcst_dt'] = pd.to_datetime(df['fcst_dt'])
            elif 'fcst_date' in df.columns:
                df['fcst_date'] = pd.to_datetime(df['fcst_date'])
            return df
        except Exception as e:
            st.error(f"'{table_name}' 데이터 로드 오류: {e}")
            return pd.DataFrame()

    @st.cache_resource
    def load_prediction_model(model_path):
        """단일 모델 로드(선택)"""
        if not os.path.exists(model_path):
            st.warning(f"모델 파일({model_path})을 찾을 수 없습니다. AI Mock-up 모드로 작동합니다.")
            return None
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"모델 로드 중 오류 발생: {e}")
            return None

    # -----------------------------------------------------------------
    # 3. 규칙 기반 정상범위(체중별 체온), 하이브리드 예측기
    # -----------------------------------------------------------------
    HEALTH_RULES = [
        (0, 30, 38.0, 39.9),
        (30, 70, 37.9, 39.8),
        (70, 1e9, 37.8, 39.7)
    ]

    def get_normal_temp_range(w):
        if pd.isna(w) or w <= 0:
            return 38.6, 39.8
        for lo, hi, tmin, tmax in HEALTH_RULES:
            if lo <= w < hi:
                return tmin, tmax
        return 38.6, 39.8

    class LSTMPredictor:
        def __init__(self, sequence_length=14):
            self.sequence_length = sequence_length
            self.model = None
            self.scaler = None

        def _maybe_imports(self):
            if keras is None or layers is None:
                raise RuntimeError("TensorFlow/Keras가 설치되어 있지 않습니다. LSTM 예측을 사용할 수 없습니다.")

        def create_sequences(self, data):
            X, y = [], []
            for i in range(len(data) - self.sequence_length):
                X.append(data[i:i + self.sequence_length])
                y.append(data[i + self.sequence_length, 0])
            return np.array(X), np.array(y)

        def build_model(self, n_features):
            self._maybe_imports()
            model = keras.Sequential([
                layers.Input(shape=(self.sequence_length, n_features)),
                layers.LSTM(64, return_sequences=True, dropout=0.0, recurrent_dropout=0.0, use_bias=True,
                            unit_forget_bias=True),
                layers.LSTM(32, return_sequences=False, dropout=0.0, recurrent_dropout=0.0, use_bias=True,
                            unit_forget_bias=True),
                layers.Dense(16, activation='relu'),
                layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model

        def train(self, df):
            try:
                from sklearn.preprocessing import StandardScaler
            except Exception:
                st.warning("scikit-learn이 없어 LSTM 학습을 건너뜁니다.")
                return None
            self._maybe_imports()
            if df.empty:
                return None
            df = df.copy()
            for a, b in [('weight', 'weight_kg'), ('feed', 'feed_intake_kg'), ('daily_gain', 'daily_gain_kg')]:
                if b in df.columns:
                    df[a] = df[b]
            features = ['weight', 'daily_gain', 'feed']
            if any(c not in df.columns for c in features):
                return None
            data_all = df[features].values
            self.scaler = StandardScaler().fit(data_all)
            df_scaled = df.copy()
            df_scaled[features] = self.scaler.transform(data_all)
            X_list, y_list = [], []
            if 'pig_id' in df_scaled.columns:
                for _, g in df_scaled.groupby('pig_id', sort=True):
                    arr = g.sort_values('day')[features].values
                    if len(arr) > self.sequence_length:
                        Xp, yp = self.create_sequences(arr)
                        if len(Xp) > 0:
                            X_list.append(Xp);
                            y_list.append(yp)
            else:
                arr = df_scaled[features].values
                Xp, yp = self.create_sequences(arr)
                if len(Xp) > 0:
                    X_list.append(Xp);
                    y_list.append(yp)
            if not X_list:
                return None
            X = np.concatenate(X_list, axis=0);
            y = np.concatenate(y_list, axis=0)
            self.model = self.build_model(n_features=len(features))
            cb = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
            self.model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2, callbacks=cb, verbose=0)
            return self.model

        def predict_future_weights(self, recent_data, n_days=60):
            if self.model is None or self.scaler is None:
                return None
            df = recent_data.copy()
            for a, b in [('weight', 'weight_kg'), ('feed', 'feed_intake_kg'), ('daily_gain', 'daily_gain_kg')]:
                if b in df.columns:
                    df[a] = df[b]
            features = ['weight', 'daily_gain', 'feed']
            if any(c not in df.columns for c in features) or len(df) < self.sequence_length:
                return None
            seq = df[features].values[-self.sequence_length:]
            seq_scaled = self.scaler.transform(seq)
            preds = []
            for _ in range(n_days):
                Xp = seq_scaled.reshape(1, self.sequence_length, -1)
                next_w_scaled = self.model.predict(Xp, verbose=0)[0, 0]
                temp = np.zeros((1, len(features)));
                temp[0, 0] = next_w_scaled
                next_w = self.scaler.inverse_transform(temp)[0, 0]
                preds.append(next_w)
                next_gain = next_w - seq[-1, 0]
                next_feed = next_w * 0.035
                next_point = np.array([[next_w, next_gain, next_feed]])
                next_point_scaled = self.scaler.transform(next_point)
                seq_scaled = np.vstack([seq_scaled[1:], next_point_scaled])
                seq = np.vstack([seq[1:], next_point])
            return preds

    class AIPredictor:
        def __init__(self, model_dir='./models'):
            self.model_dir = model_dir
            self.rf_model = None
            self.xgb_model = None
            self.rf_scaler = None
            self._load()

        def _load(self):
            try:
                rf_path = os.path.join(self.model_dir, 'random_forest_model.pkl')
                xgb_path = os.path.join(self.model_dir, 'xgboost_model.pkl')
                scaler_path = os.path.join(self.model_dir, 'random_forest_scaler.pkl')
                if os.path.exists(rf_path):
                    self.rf_model = joblib.load(rf_path)
                if os.path.exists(xgb_path) and xgb is not None:
                    self.xgb_model = joblib.load(xgb_path)
                if os.path.exists(scaler_path):
                    self.rf_scaler = joblib.load(scaler_path)
            except Exception as e:
                st.warning(f"AI 모델 로드 실패: {e}")

        def _create_feats(self, df):
            d = df.copy().sort_values('day')
            if 'weight_kg' in d.columns: d['weight'] = d['weight_kg']
            if 'feed_intake_kg' in d.columns: d['feed'] = d['feed_intake_kg']
            d['weight_lag1'] = d['weight'].shift(1)
            d['weight_lag3'] = d['weight'].shift(3)
            d['weight_lag7'] = d['weight'].shift(7)
            d['weight_rolling_mean_7'] = d['weight'].rolling(7, min_periods=1).mean()
            d['weight_rolling_std_7'] = d['weight'].rolling(7, min_periods=1).std()
            d['weight_change_1d'] = d['weight'] - d['weight_lag1']
            d['weight_change_3d'] = d['weight'] - d['weight_lag3']
            d['weight_change_7d'] = d['weight'] - d['weight_lag7']
            d['feed_weight_ratio'] = d['feed'] / d['weight']
            d['day_squared'] = d['day'] ** 2
            d['weight_squared'] = d['weight'] ** 2
            return d

        def predict_daily_gain(self, pig_data):
            d = self._create_feats(pig_data).dropna()
            if d.empty:
                return 0.65
            X_cols = [
                'weight', 'day', 'feed', 'weight_lag1', 'weight_lag3', 'weight_lag7',
                'weight_rolling_mean_7', 'weight_rolling_std_7', 'weight_change_1d',
                'weight_change_3d', 'weight_change_7d', 'feed_weight_ratio', 'day_squared', 'weight_squared'
            ]
            last = d.iloc[-1:][X_cols]
            preds = []
            if self.rf_model is not None and self.rf_scaler is not None:
                Xs = self.rf_scaler.transform(last)
                preds.append(float(self.rf_model.predict(Xs)[0]))
            if self.xgb_model is not None:
                preds.append(float(self.xgb_model.predict(last)[0]))
            return float(np.mean(preds)) if preds else 0.65

    class HybridPredictor:
        def __init__(self, target_weight=116.0):
            self.target_weight = float(target_weight)
            self.ai = AIPredictor()
            self.lstm = LSTMPredictor()

        def train_lstm(self, df_all):
            try:
                self.lstm.train(df_all)
            except Exception as e:
                st.info(f"LSTM 학습 생략: {e}")

        def predict_shipment(self, pig_hist: pd.DataFrame):
            cw = float(pig_hist['weight_kg'].iloc[-1])
            if cw >= self.target_weight:
                return {
                    'status': 'ready', 'final_days_to_shipment': 0,
                    'predicted_shipment_date': datetime.now().strftime('%Y-%m-%d')
                }
            ai_gain = float(self.ai.predict_daily_gain(pig_hist))
            remain = max(0.0, self.target_weight - cw)
            ai_days = max(1, int(np.ceil(remain / max(ai_gain, 1e-3))))
            lstm_days = None
            try:
                preds = self.lstm.predict_future_weights(pig_hist, n_days=60)
                if preds:
                    for i, w in enumerate(preds, 1):
                        if w >= self.target_weight:
                            lstm_days = i;
                            break
                    if lstm_days is None:
                        lstm_days = 60
            except Exception:
                pass
            if 'daily_gain_kg' in pig_hist.columns and pig_hist['daily_gain_kg'].notna().any():
                rg = pig_hist['daily_gain_kg'].tail(7).mean()
            else:
                rg = np.nan
            if pd.isna(rg) or rg <= 0.01:
                rg = 0.6
            stat_days = max(1, int(np.ceil(remain / rg)))
            preds = [ai_days, stat_days];
            wts = [0.4, 0.2]
            if lstm_days is not None:
                preds.insert(1, lstm_days);
                wts.insert(1, 0.4)
            else:
                wts[0] += 0.2
            wts = np.array(wts) / np.sum(wts)
            final_days = int(np.round(np.average(preds, weights=wts)))
            final_days = int(np.clip(final_days, max(1, int(remain / 1.2)), int(remain / 0.3)))
            return {
                'status': 'predicted',
                'ai_prediction_days': ai_days,
                'lstm_prediction_days': lstm_days if lstm_days is not None else 'N/A',
                'stat_prediction_days': stat_days,
                'final_days_to_shipment': final_days,
                'predicted_shipment_date': (datetime.now() + timedelta(days=final_days)).strftime('%Y-%m-%d'),
                'ai_daily_gain': round(ai_gain, 3),
                'recent_daily_gain': round(rg, 3)
            }

    @st.cache_resource
    def load_hybrid_predictor():
        try:
            return HybridPredictor(target_weight=116.0)
        except Exception as e:
            st.error(f"하이브리드 예측기 로드 실패: {e}")
            return None

    # -----------------------------------------------------------------
    # 4. 라우팅 상태
    # -----------------------------------------------------------------
    def set_detail_view(chamber_id, chamber_no):
        st.session_state.view_mode = 'detail'
        st.session_state.selected_chamber_id = chamber_id
        st.session_state.selected_chamber_no = chamber_no


    def set_overview_view():
        st.session_state.view_mode = 'overview'
        st.session_state.selected_chamber_id = None
        st.session_state.selected_chamber_no = None
        st.session_state.pop('prediction_results', None)

    if 'view_mode' not in st.session_state:
        set_overview_view()

    # -----------------------------------------------------------------
    # 5. 데이터 로드
    # -----------------------------------------------------------------
    engine = init_db_connection()

    sensor_df_all = load_data_from_db(engine, 'Chamber_Logs', limit=20000)
    equipment_df_all = load_data_from_db(engine, 'Equipment_Logs', limit=20000)
    pig_log_df_all = load_data_from_db(engine, 'Pig_Logs', limit='None', order_by_col='timestamp')
    chambers_df = load_data_from_db(engine, 'Chambers', limit='None', order_by_col=None)
    pigs_df = load_data_from_db(engine, 'Pigs', limit='None', order_by_col=None)
    mid_land_fcst_df = load_data_from_db(engine, "mid_land_fcst", limit=7, order_by_col="fcst_date")
    weather_ultra_fcst_df = load_data_from_db(engine, "weather_ultra_fcst", limit=48, order_by_col="fcst_dt")

    if not weather_ultra_fcst_df.empty:
        weather_ultra_fcst_df.columns = [c.upper() for c in weather_ultra_fcst_df.columns]

    if 'weight_kg' in pig_log_df_all.columns:
        pig_log_df_all['weight_kg'] = pd.to_numeric(pig_log_df_all['weight_kg'], errors='coerce')

    # (선택) 챔버 1,2에서 20마리 샘플링 + 나머지 챔버는 전체 유지
    CHAMBER_IDS_TO_SAMPLE = [1, 2]
    PIGS_PER_CHAMBER = 20
    if not pigs_df.empty and not pig_log_df_all.empty:
        try:
            rng = np.random.default_rng(42)
            pigs_sel = []
            for cid in CHAMBER_IDS_TO_SAMPLE:
                pig_ids = pigs_df[pigs_df['chamber_id'] == cid]['pig_id'].unique()
                k = min(len(pig_ids), PIGS_PER_CHAMBER)
                if k > 0:
                    pigs_sel.append(rng.choice(pig_ids, size=k, replace=False))
            pigs_keep = pigs_df[~pigs_df['chamber_id'].isin(CHAMBER_IDS_TO_SAMPLE)]['pig_id'].unique()
            final_ids = np.concatenate(pigs_sel + [pigs_keep]) if pigs_sel else pigs_keep
            pigs_df = pigs_df[pigs_df['pig_id'].isin(final_ids)].copy()
            pig_log_df_all = pig_log_df_all[pig_log_df_all['pig_id'].isin(final_ids)].copy()
        except Exception as e:
            st.warning(f"샘플링 생략: {e}")

    # =================================================================
    # A. OVERVIEW
    # =================================================================
    if st.session_state.view_mode == "overview":
        st.title("🐷 스마트 축사 현황 (전체 맵)")

        with st.container(border=True):
            st.subheader("AICU 총괄 요약")
            cols = st.columns(5)

            if not pig_log_df_all.empty:
                cols[0].metric("총 사육 두수", f"{pig_log_df_all['pig_id'].nunique()} 마리")
            else:
                cols[0].metric("총 사육 두수", "N/A")

            # 체중별 정상범위 기반 '주의' 개체
            if not pig_log_df_all.empty:
                try:
                    pig_log_df_all['temp_rectal'] = pd.to_numeric(pig_log_df_all['temp_rectal'], errors='coerce')
                    pig_log_df_all['breath_rate'] = pd.to_numeric(pig_log_df_all['breath_rate'], errors='coerce')
                except Exception:
                    pass
                valid = pig_log_df_all.dropna(subset=['temp_rectal', 'breath_rate', 'weight_kg'])
                if not valid.empty:
                    latest = valid.loc[valid.groupby("pig_id")["timestamp"].idxmax()].copy()
                    latest['tmin'], latest['tmax'] = zip(*latest['weight_kg'].apply(get_normal_temp_range))
                    bmin, bmax = 55, 70
                    warn = latest[(latest["temp_rectal"] < latest["tmin"]) |
                                  (latest["temp_rectal"] > latest["tmax"]) |
                                  (latest["breath_rate"] < bmin) |
                                  (latest["breath_rate"] > bmax)]
                    cols[1].metric("총 '주의' 개체 수", f"{len(warn)} 마리")
                else:
                    cols[1].metric("총 '주의' 개체 수", "N/A")
            else:
                cols[1].metric("총 '주의' 개체 수", "N/A")

            if not weather_ultra_fcst_df.empty and {"T1H", "REH"}.issubset(weather_ultra_fcst_df.columns):
                latest_weather = weather_ultra_fcst_df.iloc[0]
                cols[2].metric("현재 외부 온도", f"{latest_weather.get('T1H', 0):.1f} °C")
                cols[3].metric("현재 외부 습도", f"{latest_weather.get('REH', 0):.1f} %")
            else:
                cols[2].metric("현재 외부 온도", "N/A")
                cols[3].metric("현재 외부 습도", "N/A")

            if not mid_land_fcst_df.empty and {"pop_am", "pop_pm"}.issubset(mid_land_fcst_df.columns):
                today = mid_land_fcst_df.iloc[0]
                pa, pp = today.get("pop_am", 0), today.get("pop_pm", 0)
                cols[4].metric("오전/오후 강수", f"{pa}% / {pp}%")
                if (pa or 0) > 70 or (pp or 0) > 70:
                    st.warning("🚨 강수 확률 70% 이상! 환기/습도 관리 유의")
            else:
                cols[4].metric("강수 확률", "N/A")

        st.divider()
        st.subheader("챔버별 현황 (클릭하여 드릴다운)")

        if chambers_df.empty:
            st.error("챔버 정보를 찾을 수 없습니다.")
        else:
            grid_cols = st.columns(2)

            valid_logs_all = pd.DataFrame()
            if not pig_log_df_all.empty:
                valid_logs_all = pig_log_df_all.dropna(subset=['temp_rectal', 'breath_rate', 'weight_kg'])
                if not valid_logs_all.empty:
                    valid_logs_all = valid_logs_all.loc[valid_logs_all.groupby("pig_id")["timestamp"].idxmax()].copy()
                    valid_logs_all['tmin'], valid_logs_all['tmax'] = zip(
                        *valid_logs_all['weight_kg'].apply(get_normal_temp_range))

            for i, row in chambers_df.iterrows():
                chamber_id = row['chamber_id'];
                chamber_no = row['chamber_no']
                current_col = grid_cols[i % 2]
                warn_count = 0
                if not pigs_df.empty and not valid_logs_all.empty:
                    ids = pigs_df[pigs_df['chamber_id'] == chamber_id]['pig_id']
                    latest = valid_logs_all[valid_logs_all['pig_id'].isin(ids)]
                    bmin, bmax = 55, 70
                    warning = latest[(latest["temp_rectal"] < latest["tmin"]) |
                                     (latest["temp_rectal"] > latest["tmax"]) |
                                     (latest["breath_rate"] < bmin) |
                                     (latest["breath_rate"] > bmax)]
                    warn_count = len(warning)

                with current_col.container(border=True):
                    if warn_count >= 5:
                        st.error(f"🚨 {chamber_no}번 챔버 (주의!)")
                    else:
                        st.subheader(f"{chamber_no}번 챔버")

                    c1, c2 = st.columns(2)
                    d = sensor_df_all[sensor_df_all['chamber_id'] == chamber_id]
                    if not d.empty and "temperature" in d.columns:
                        c1.metric("현재 온도", f"{d.iloc[0]['temperature']:.1f} °C")
                    else:
                        c1.metric("현재 온도", "N/A")
                    c2.metric("건강 '주의' 개체", f"{warn_count} 마리")

                    st.button(
                        f"{chamber_no}번 챔버 상세 정보 보기",
                        key=f"btn_detail_{chamber_id}",
                        on_click=set_detail_view,
                        args=(chamber_id, chamber_no)
                    )

        # ---- 전체 에너지 사용량 분석 (모든 챔버) ----
        st.header("전체 에너지 사용량 분석 (모든 챔버)")
        if not equipment_df_all.empty:
            min_d = equipment_df_all['timestamp'].min().date()
            max_d = equipment_df_all['timestamp'].max().date()
            dr = st.date_input("조회 기간을 선택하세요:",
                               value=(min_d, max_d),
                               min_value=min_d, max_value=max_d,
                               key="energy_date_selector_overview")
            data = pd.DataFrame();
            start_str, end_str = min_d.isoformat(), max_d.isoformat()
            if len(dr) == 2:
                s = pd.to_datetime(dr[0]);
                e = pd.to_datetime(dr[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                start_str, end_str = dr[0].isoformat(), dr[1].isoformat()
                data = equipment_df_all[(equipment_df_all['timestamp'] >= s) & (equipment_df_all['timestamp'] <= e)]
            if data.empty:
                st.info("선택된 기간에 해당하는 에너지 데이터가 없습니다.")
            else:
                period = data.groupby('equipment_type')['power_usage_wh'].sum() / 1000
                fig = px.bar(period, title="장비별 기간 내 사용량 (kWh)",
                             labels={'value': '사용량 (kWh)', 'equipment_type': '장비 종류'})
                st.plotly_chart(fig, use_container_width=True)

                @st.cache_data
                def to_csv(df):
                    return df.to_csv(index=False, encoding='utf-8-sig')

                st.download_button(
                    label=f"📈 전체 에너지 로그 다운로드",
                    data=to_csv(data),
                    file_name=f"energy_logs_ALL_{start_str}_to_{end_str}.csv",
                    mime="text/csv",
                )
        else:
            st.warning("에너지 사용량 데이터가 없습니다.")

        # ---- 일일 날씨(시간별) ----
        st.header("🌦️ 일일 날씨")
        need_cols = {"FCST_DT", "T1H", "REH", "RN1", "SKY", "PTY"}
        if not weather_ultra_fcst_df.empty and need_cols.issubset(weather_ultra_fcst_df.columns):
            wd = weather_ultra_fcst_df.set_index("FCST_DT")
            t1, t2, t3 = st.tabs(["🌡️ 기온 (T1H)", "💧 습도 (REH)", "☔ 시간당 강수량 (RN1)"])
            with t1:
                st.plotly_chart(px.line(wd, y='T1H', title='시간별 외부 기온'), use_container_width=True)
            with t2:
                st.plotly_chart(px.line(wd, y='REH', title='시간별 외부 습도'), use_container_width=True)
            with t3:
                st.plotly_chart(px.bar(wd, y='RN1', title='시간별 강수량'), use_container_width=True)
            st.info(f"현재 SKY 코드: {weather_ultra_fcst_df.iloc[0].get('SKY', -1)} (1:맑음, 3:구름많음, 4:흐림)")
        else:
            st.warning("시간별 상세 날씨(weather_ultra_fcst) 데이터가 부족합니다.")

        # ---- 주간 날씨(이모지) ----
        st.subheader("🗓️ 주간 날씨 요약 (기상청 DB)")
        need = ["fcst_date", "wf_am", "pop_am", "wf_pm", "pop_pm", "tmin", "tmax"]
        if not mid_land_fcst_df.empty and all(c in mid_land_fcst_df.columns for c in need):
            display_df = mid_land_fcst_df[need].copy().sort_values("fcst_date")
            display_df['fcst_date'] = display_df['fcst_date'].dt.strftime('%m월 %d일 (%a)')
            weather_emoji = {
                "맑음": "☀️", "구름많음": "🌥️", "흐림": "☁️",
                "비": "🌧️", "눈": "❄️", "비/눈": "🌨️", "소나기": "🌦️"
            }
            display_df['오전 날씨'] = display_df['wf_am'].replace(weather_emoji)
            display_df['오후 날씨'] = display_df['wf_pm'].replace(weather_emoji)
            display_df['일일 강수 확률(%)'] = display_df[['pop_am', 'pop_pm']].max(axis=1).astype(int)
            final_cols = ["fcst_date", "일일 강수 확률(%)", "오전 날씨", "오후 날씨", "tmin", "tmax"]
            display_df = display_df[final_cols].rename(columns={
                "fcst_date": "날짜", "tmin": "최저 기온(°C)", "tmax": "최고 기온(°C)"
            })
            st.dataframe(display_df.set_index("날짜"), use_container_width=True)
        else:
            st.warning("주간 날씨 요약(mid_land_fcst) 데이터가 없습니다.")

    # =================================================================
    # B. DETAIL
    # =================================================================
    elif st.session_state.view_mode == "detail":
        st.button("◀ 전체 맵으로 돌아가기", on_click=set_overview_view)
        selected_id = st.session_state.selected_chamber_id
        selected_no = st.session_state.selected_chamber_no
        st.title(f"🐷 {selected_no}번 챔버 상세 정보")

        sensor_df_filtered = sensor_df_all[sensor_df_all['chamber_id'] == selected_id]
        equipment_df_filtered = equipment_df_all[equipment_df_all['chamber_id'] == selected_id]

        pig_log_df_filtered = pd.DataFrame()
        if not pigs_df.empty:
            ids = pigs_df[pigs_df['chamber_id'] == selected_id]['pig_id']
            pig_log_df_filtered = pig_log_df_all[pig_log_df_all['pig_id'].isin(ids)].copy()

        st.divider()
        st.header("📈 현재 챔버 상황")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 환경 센서 (Chamber_Logs)")
            if not sensor_df_filtered.empty:
                latest = sensor_df_filtered.iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("온도", f"{latest['temperature']:.1f} °C")
                c2.metric("습도", f"{latest['humidity']:.1f} %")
                c3.metric("CO2", f"{latest['co2']:.0f} ppm")
                min_d = sensor_df_filtered['timestamp'].min().date()
                max_d = sensor_df_filtered['timestamp'].max().date()
                dr = st.date_input("조회 기간을 선택하세요:", value=(min_d, max_d),
                                   min_value=min_d, max_value=max_d,
                                   key=f"date_selector_{selected_id}")
                chart_df = pd.DataFrame()
                if len(dr) == 2:
                    s = pd.to_datetime(dr[0]);
                    e = pd.to_datetime(dr[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                    chart_df = sensor_df_filtered[
                        (sensor_df_filtered['timestamp'] >= s) & (sensor_df_filtered['timestamp'] <= e)]
                if chart_df.empty:
                    st.info("선택된 기간 데이터 없음")
                else:
                    t1, t2, t3 = st.tabs(["🌡️ 온도", "💧 습도", "💨 CO2"])
                    with t1:
                        st.plotly_chart(px.line(chart_df, x='timestamp', y='temperature', title='온도 추이'),
                                        use_container_width=True)
                    with t2:
                        st.plotly_chart(px.line(chart_df, x='timestamp', y='humidity', title='습도 추이'),
                                        use_container_width=True)
                    with t3:
                        st.plotly_chart(px.line(chart_df, x='timestamp', y='co2', title='CO2 추이'),
                                        use_container_width=True)
            else:
                st.warning("센서 데이터를 찾을 수 없습니다.")

        with col2:
            st.subheader("❤️ 돼지 건강 상태 (Pig_Logs)")
            if not pig_log_df_filtered.empty:
                try:
                    pig_log_df_filtered['temp_rectal'] = pd.to_numeric(pig_log_df_filtered['temp_rectal'],
                                                                       errors='coerce')
                    pig_log_df_filtered['breath_rate'] = pd.to_numeric(pig_log_df_filtered['breath_rate'],
                                                                       errors='coerce')
                except Exception as e:
                    st.warning(f"타입 변환 오류: {e}")
                valid = pig_log_df_filtered.dropna(subset=['temp_rectal', 'breath_rate', 'weight_kg'])
                if not valid.empty:
                    latest = valid.loc[valid.groupby('pig_id')['timestamp'].idxmax()].copy()
                    latest['tmin'], latest['tmax'] = zip(*latest['weight_kg'].apply(get_normal_temp_range))
                    bmin, bmax = 55, 70
                    warn = latest[(latest['temp_rectal'] < latest["tmin"]) |
                                  (latest['temp_rectal'] > latest["tmax"]) |
                                  (latest['breath_rate'] < bmin) |
                                  (latest['breath_rate'] > bmax)]
                    st.metric("건강 '주의' 개체 수", f"{len(warn)} 마리")
                    if len(warn) > 0:
                        with st.expander("'주의' 개체 목록 보기"):
                            def reason(r):
                                rs = []
                                if r['temp_rectal'] < r['tmin']:
                                    rs.append(f"온도 낮음 ({r['temp_rectal']:.1f}°C)")
                                elif r['temp_rectal'] > r['tmax']:
                                    rs.append(f"온도 높음 ({r['temp_rectal']:.1f}°C)")
                                if r['breath_rate'] < bmin:
                                    rs.append(f"호흡 느림 ({r['breath_rate']:.0f}회)")
                                elif r['breath_rate'] > bmax:
                                    rs.append(f"호흡 빠름 ({r['breath_rate']:.0f}회)")
                                return ", ".join(rs)

                            w = warn.copy();
                            w['주의 원인'] = w.apply(reason, axis=1)
                            st.dataframe(w[["pig_id", "weight_kg", "temp_rectal", "breath_rate", "주의 원인"]],
                                         use_container_width=True)
                else:
                    st.warning("유효한 건강 데이터가 없습니다.")
            else:
                st.warning("돼지 로그 데이터를 찾을 수 없습니다.")

        # --- (NEW) 챔버 돼지 로그 CSV 다운로드 ---
        st.divider()
        st.subheader("⬇️ 챔버 돼지 로그 CSV 다운로드")

        if not pig_log_df_filtered.empty:
            try:
                _min_d = pig_log_df_filtered['timestamp'].min().date()
                _max_d = pig_log_df_filtered['timestamp'].max().date()
                _dr = st.date_input(
                    "다운로드 범위 선택",
                    value=(_min_d, _max_d),
                    min_value=_min_d, max_value=_max_d,
                    key=f"pig_download_dates_{selected_id}"
                )

                _all_cols = list(pig_log_df_filtered.columns)
                _default_cols = [c for c in ["timestamp","pig_id","weight_kg","temp_rectal","breath_rate","daily_gain_kg"] if c in _all_cols]
                _sel_cols = st.multiselect(
                    "다운로드 항목 선택",
                    options=_all_cols,
                    default=_default_cols if _default_cols else _all_cols[:8],
                    key=f"pig_download_cols_{selected_id}"
                )

                _enc = st.radio(
                    "파일 인코딩 설정",
                    options=["utf-8-sig","cp949","euc-kr"],
                    index=0,
                    horizontal=True,
                    key=f"pig_download_enc_{selected_id}"
                )

                _data = pd.DataFrame()
                _start_str, _end_str = _min_d.isoformat(), _max_d.isoformat()
                if len(_dr) == 2:
                    _s = pd.to_datetime(_dr[0])
                    _e = pd.to_datetime(_dr[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                    _start_str, _end_str = _dr[0].isoformat(), _dr[1].isoformat()
                    _data = pig_log_df_filtered[(pig_log_df_filtered['timestamp'] >= _s) & (pig_log_df_filtered['timestamp'] <= _e)].copy()

                if _data.empty:
                    st.info("선택된 기간에 돼지 로그가 없습니다.")
                else:
                    if _sel_cols:
                        _data = _data[[c for c in _sel_cols if c in _data.columns]]

                    with st.expander("미리보기 (상위 20행)"):
                        st.dataframe(_data.sort_values("timestamp").head(20), use_container_width=True)

                    @st.cache_data(show_spinner=False)
                    def _to_csv_pig(df, enc):
                        return df.to_csv(index=False, encoding=enc)

                    st.download_button(
                        label=f"🐖 CSV 다운로드 (챔버 {selected_no}, {_start_str} ~ {_end_str})",
                        data=_to_csv_pig(_data.sort_values("timestamp"), _enc),
                        file_name=f"pig_logs_{selected_no}ch_{_start_str}_to_{_end_str}.csv",
                        mime="text/csv",
                        key=f"pig_download_btn_{selected_id}"
                    )
            except Exception as e:
                st.warning(f"CSV 생성 중 오류: {e}")
        else:
            st.info("이 챔버에 돼지 로그 데이터가 없습니다.")


        st.divider()
        # --- 출하 & 에너지 탭 ---
        st.header("🐖 출하 및 에너지 분석")
        tab1, tab2 = st.tabs(["출하 날짜 예측(하이브리드)", "에너지 사용량 분석"])

        with tab1:
            predictor = load_hybrid_predictor()

            def clear_prediction_results():
                st.session_state.pop('prediction_results', None)

            target_weight = st.number_input("목표 출하 체중(kg)", min_value=80.0, value=116.0, step=1.0)
            if predictor is not None:
                predictor.target_weight = float(target_weight)

            if 'prediction_results' not in st.session_state:
                st.session_state.prediction_results = None

            if not pig_log_df_filtered.empty and predictor is not None:
                feed_df = sensor_df_filtered[['timestamp',
                                              'feed_volume']].dropna() if 'feed_volume' in sensor_df_filtered.columns else pd.DataFrame(
                    columns=['timestamp', 'feed_volume'])
                merged = pd.merge(pig_log_df_filtered, feed_df, on="timestamp", how="left")
                merged = merged.rename(columns={'feed_volume': 'feed_intake_kg'})
                if 'day' not in merged.columns:
                    merged = merged.sort_values(by=['pig_id', 'timestamp'])
                    merged['day'] = merged.groupby('pig_id')['timestamp'].transform(lambda x: (x - x.min()).dt.days)
                if 'daily_gain_kg' not in merged.columns:
                    merged['weight_lag1'] = merged.groupby('pig_id')['weight_kg'].shift(1)
                    merged['daily_gain_kg'] = (merged['weight_kg'] - merged['weight_lag1']).fillna(0.6)

                logs_w = merged.dropna(subset=['weight_kg'])
                if not logs_w.empty:
                    latest_w = logs_w.loc[logs_w.groupby("pig_id")["timestamp"].idxmax()]
                    ready = latest_w[latest_w['weight_kg'] >= target_weight]
                    c1, c2 = st.columns(2)
                    c1.metric(f"현재 {target_weight}kg 이상", f"{len(ready)} 마리")
                    below = latest_w[latest_w['weight_kg'] < target_weight]
                    c2.metric("출하 예측 대상", f"{len(below)} 마리")
                    st.divider()

                    st.subheader(f"🐷 {target_weight}kg 도달 날짜 예측 (AI 하이브리드)")
                    if st.button(f"🐷 {len(below)}마리 출하 예측 실행하기", key=f"predict_btn_{selected_id}"):
                        if not below.empty:
                            results = []
                            with st.spinner(f"{len(below)}마리 출하 예측 계산 중..."):
                                predictor.train_lstm(logs_w)
                                for _, rep in below.iterrows():
                                    pid = rep['pig_id']
                                    hist = logs_w[logs_w['pig_id'] == pid]
                                    res = predictor.predict_shipment(hist)
                                    results.append({
                                        '돼지 ID': int(pid),
                                        '현재 체중(kg)': round(float(rep['weight_kg']), 1),
                                        '남은 일수(일)': int(res['final_days_to_shipment']),
                                        '예상 출하 날짜': res['predicted_shipment_date']
                                    })
                            st.session_state.prediction_results = pd.DataFrame(results).sort_values('남은 일수(일)')
                        else:
                            st.success(f"모든 개체가 목표 체중({target_weight}kg) 이상입니다.")
                            st.session_state.prediction_results = pd.DataFrame()

                    if st.session_state.prediction_results is not None:
                        result_df = st.session_state.prediction_results
                        if not result_df.empty:
                            fastest = result_df.iloc[0]
                            st.metric(f"가장 빠른 예상 출하일 (ID: {fastest['돼지 ID']})",
                                      f"{fastest['예상 출하 날짜']}", f"{fastest['남은 일수(일)']}일 남음")
                            with st.expander("전체 개체별 예상 출하일 보기 (빠른 순)"):
                                st.dataframe(result_df.set_index('돼지 ID'), use_container_width=True)
                        else:
                            st.success("예측 대상이 없거나 모두 목표 체중 이상입니다.")
                    else:
                        st.info("AI 예측을 실행하려면 버튼을 클릭하세요.")
                else:
                    st.warning("이 챔버에는 유효한 체중 데이터가 없습니다.")
            else:
                st.warning("몸무게 데이터가 없거나 AI 예측기를 로드하지 못했습니다.")

        with tab2:
            if not equipment_df_filtered.empty:
                min_d = equipment_df_filtered['timestamp'].min().date()
                max_d = equipment_df_filtered['timestamp'].max().date()
                dr = st.date_input("조회 기간을 선택하세요:", value=(min_d, max_d),
                                   min_value=min_d, max_value=max_d,
                                   key=f"energy_date_selector_{selected_id}")
                data = pd.DataFrame();
                start_str, end_str = min_d.isoformat(), max_d.isoformat()
                if len(dr) == 2:
                    s = pd.to_datetime(dr[0]);
                    e = pd.to_datetime(dr[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                    start_str, end_str = dr[0].isoformat(), dr[1].isoformat()
                    data = equipment_df_filtered[
                        (equipment_df_filtered['timestamp'] >= s) & (equipment_df_filtered['timestamp'] <= e)]
                if data.empty:
                    st.info("선택된 기간에 해당하는 에너지 데이터가 없습니다.")
                else:
                    st.subheader(f"기간 내 장비별 사용량 ({start_str} ~ {end_str})")
                    period = data.groupby('equipment_type')['power_usage_wh'].sum() / 1000
                    fig = px.bar(period, title="장비별 기간 내 사용량 (kWh)",
                                 labels={'value': '사용량 (kWh)', 'equipment_type': '장비 종류'})
                    st.plotly_chart(fig, use_container_width=True)

                    @st.cache_data
                    def to_csv(df):
                        return df.to_csv(index=False, encoding='utf-8-sig')

                    st.download_button(
                        label=f"📈 기간({start_str}~{end_str}) 로그 다운로드",
                        data=to_csv(data),
                        file_name=f"energy_logs_{selected_no}ch_{start_str}_to_{end_str}.csv",
                        mime="text/csv",
                    )

    # -----------------------------------------------------------------
    # FAB logout link
    # -----------------------------------------------------------------
    st.markdown('<a class="fab-logout" href="?logout=1">로그아웃</a>', unsafe_allow_html=True)


if __name__ == '__main__':
    render()

