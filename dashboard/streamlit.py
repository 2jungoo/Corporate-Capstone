import streamlit as st
import pandas as pd
import sqlalchemy as sa
import requests
from datetime import datetime
import plotly.express as px
import numpy as np
import os
import joblib
import json
import os
import warnings
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

target_weight = 116 # (대시보드 기준 80kg로 수정)

# 1. 체중(kg)대별 정상 체온(°C) 범위 규칙
HEALTH_RULES = [
    (0, 30, 38.0, 39.9),   # 0-30kg : 38.7°C ~ 39.9°C
    (30, 70, 37.9, 39.8),  # 30-70kg: 38.6°C ~ 39.8°C
    (70, 1e9, 37.8, 39.7)  # 70kg+  : 38.5°C ~ 39.7°C
]

# 2. 체중(w)을 받아, 정상 범위(tmin, tmax)를 반환하는 함수
def get_normal_temp_range(w):
    if pd.isna(w) or w <= 0: # 체중값이 없으면 기본값 반환
        return 38.6, 39.8
    for lo, hi, tmin, tmax in HEALTH_RULES:
        if lo <= w < hi:
            return tmin, tmax
    return 38.6, 39.8 # (기본값)

class DummyDataGenerator:
    def __init__(self, pattern_file='./models/growth_patterns_ai.json'):
        self.pattern_file = pattern_file
        self.target_weight = target_weight
        self.patterns = None
        self.load_patterns()

    def load_patterns(self):
        print("\n" + "=" * 80)
        print("학습된 패턴 로딩 중...")
        print("=" * 80)
        if not os.path.exists(self.pattern_file):
            print("패턴 파일이 없어 기본값 사용")
            self.patterns = self.get_default_patterns()
        else:
            with open(self.pattern_file, 'r', encoding='utf-8') as f:
                self.patterns = json.load(f)
        print("패턴 로드 완료")
        return self.patterns

    def get_default_patterns(self):
        return {
            'overall': {
                'mean_daily_gain': 0.65,
                'std_daily_gain': 0.15,
                'min_weight': 20.0,
                'max_weight': 110.0
            },
            'weight_bins': {
                '0-20kg': {'mean_daily_gain': 0.45, 'std_daily_gain': 0.10},
                '20-40kg': {'mean_daily_gain': 0.65, 'std_daily_gain': 0.12},
                '40-60kg': {'mean_daily_gain': 0.75, 'std_daily_gain': 0.10},
                '60-80kg': {'mean_daily_gain': 0.80, 'std_daily_gain': 0.10},
                '80-100kg': {'mean_daily_gain': 0.70, 'std_daily_gain': 0.12},
                '100kg+': {'mean_daily_gain': 0.55, 'std_daily_gain': 0.15}
            }
        }

    def get_daily_gain_for_weight(self, weight, day):
        weight_bins = {
            '0-20kg': (0, 20), '20-40kg': (20, 40), '40-60kg': (40, 60),
            '60-80kg': (60, 80), '80-100kg': (80, 100), '100kg+': (100, 200)
        }
        for bin_name, (min_w, max_w) in weight_bins.items():
            if min_w <= weight < max_w:
                if bin_name in self.patterns['weight_bins']:
                    bin_data = self.patterns['weight_bins'][bin_name]
                    mean_gain = bin_data['mean_daily_gain']
                    std_gain = bin_data.get('std_daily_gain', mean_gain * 0.15)
                    daily_gain = np.random.normal(mean_gain, std_gain * 0.3)
                    growth_factor = 1.0
                    if day < 30:
                        growth_factor = 0.7 + (day / 30) * 0.3
                    elif day > 120:
                        growth_factor = max(0.6, 1.0 - (day - 120) / 200)
                    daily_gain *= growth_factor
                    daily_gain = np.clip(daily_gain, 0.2, 1.2)
                    return daily_gain
        return 0.6

    def generate_pig_data(self, pig_id, start_weight=None, n_days=60):
        if start_weight is None:
            start_weight = np.random.uniform(20, 30)
        data = []
        current_weight = start_weight
        for day in range(n_days):
            daily_gain = self.get_daily_gain_for_weight(current_weight, day)
            current_weight = current_weight + abs(daily_gain)
            measured_weight = current_weight + np.random.normal(0, 0.1)
            feed_intake = current_weight * 0.035 * np.random.uniform(0.95, 1.05)
            feed_intake = max(0.5, feed_intake)
            data.append({
                'pig_id': pig_id,
                'day': day,
                'weight_kg': round(measured_weight, 2),
                'daily_gain_kg': round(daily_gain, 3),
                'feed_intake_kg': round(feed_intake, 2),
                'temperature_c': round(22 + np.random.normal(0, 2), 1),
                'humidity_percent': round(65 + np.random.normal(0, 5), 1),
                'date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d')
            })
        return pd.DataFrame(data)

    def generate_dummy_dataset(self, n_pigs=10, n_days=60):
        print(f"\n{n_pigs}마리 돼지 {n_days}일치 더미 데이터 생성 중...")
        all_data = []
        for pig_id in range(1, n_pigs + 1):
            pig_data = self.generate_pig_data(pig_id, n_days=n_days)
            all_data.append(pig_data)
        dataset = pd.concat(all_data, ignore_index=True)
        print(f"총 {len(dataset)}건 생성 완료")
        return dataset


class LSTMPredictor:
    def __init__(self, sequence_length=14):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length, 0])
        return np.array(X), np.array(y)

    def build_model(self, n_features):
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, n_features)),
            layers.LSTM(64, return_sequences=True, dropout=0.0, recurrent_dropout=0.0, use_bias=True, unit_forget_bias=True),
            layers.LSTM(32, return_sequences=False, dropout=0.0, recurrent_dropout=0.0, use_bias=True, unit_forget_bias=True),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train(self, df):
        print("\nLSTM 모델 학습 중...")
        df = df.copy()
        if 'weight_kg' in df.columns:
            df['weight'] = df['weight_kg']
        if 'feed_intake_kg' in df.columns:
            df['feed'] = df['feed_intake_kg']
        if 'daily_gain_kg' in df.columns:
            df['daily_gain'] = df['daily_gain_kg']
        features = ['weight', 'daily_gain', 'feed']
        missing = [f for f in features if f not in df.columns]
        if missing:
            print(f"필요한 feature 없음: {missing} - LSTM 학습 건너뜀")
            return None
        data_all = df[features].values
        data_scaled_all = self.scaler.fit_transform(data_all)
        df_scaled = df.copy()
        df_scaled[features] = data_scaled_all
        X_list, y_list = [], []
        if 'pig_id' in df_scaled.columns:
            groups = df_scaled.groupby('pig_id', sort=True)
            for _, g in groups:
                arr = g.sort_values('day')[features].values
                if len(arr) > self.sequence_length:
                    Xp, yp = self.create_sequences(arr)
                    if len(Xp) > 0:
                        X_list.append(Xp)
                        y_list.append(yp)
        else:
            Xp, yp = self.create_sequences(df_scaled[features].values)
            if len(Xp) > 0:
                X_list.append(Xp); y_list.append(yp)
        if not X_list:
            print("학습 데이터 부족 - LSTM 학습 건너뜀")
            return None
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        self.model = self.build_model(n_features=len(features))
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = self.model.fit(
            X, y,
            epochs=50,
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )
        final_loss = history.history['val_loss'][-1]
        final_mae = history.history['mae'][-1]
        print(f"LSTM 학습 완료 - ValLoss: {final_loss:.4f}, MAE: {final_mae:.4f}")
        return self.model

    def predict_future_weights(self, recent_data, n_days=60):
        if self.model is None:
            return None
        recent_data = recent_data.copy()
        if 'weight_kg' in recent_data.columns:
            recent_data['weight'] = recent_data['weight_kg']
        if 'feed_intake_kg' in recent_data.columns:
            recent_data['feed'] = recent_data['feed_intake_kg']
        if 'daily_gain_kg' in recent_data.columns:
            recent_data['daily_gain'] = recent_data['daily_gain_kg']
        features = ['weight', 'daily_gain', 'feed']
        missing_features = [f for f in features if f not in recent_data.columns]
        if missing_features:
            return None
        data = recent_data[features].values
        if len(data) < self.sequence_length:
            return None
        sequence = data[-self.sequence_length:]
        sequence_scaled = self.scaler.transform(sequence)
        predictions = []
        for _ in range(n_days):
            X_pred = sequence_scaled.reshape(1, self.sequence_length, -1)
            next_weight_scaled = self.model.predict(X_pred, verbose=0)[0, 0]
            temp_data = np.zeros((1, len(features)))
            temp_data[0, 0] = next_weight_scaled
            next_weight = self.scaler.inverse_transform(temp_data)[0, 0]
            predictions.append(next_weight)
            next_daily_gain = next_weight - sequence[-1, 0]
            next_feed = next_weight * 0.035
            next_point = np.array([[next_weight, next_daily_gain, next_feed]])
            next_point_scaled = self.scaler.transform(next_point)
            sequence_scaled = np.vstack([sequence_scaled[1:], next_point_scaled])
            sequence = np.vstack([sequence[1:], next_point])
        return predictions


class AIPredictor:
    def __init__(self, model_dir='./models'):
        self.model_dir = model_dir
        self.rf_model = None
        self.xgb_model = None
        self.rf_scaler = None
        self.load_models()

    def load_models(self):
        print("\nAI 모델 로딩 중...")
        try:
            rf_path = os.path.join(self.model_dir, 'random_forest_model.pkl')
            xgb_path = os.path.join(self.model_dir, 'xgboost_model.pkl')
            scaler_path = os.path.join(self.model_dir, 'random_forest_scaler.pkl')
            if os.path.exists(rf_path):
                self.rf_model = joblib.load(rf_path)
                print("Random Forest 로드")
            if os.path.exists(xgb_path):
                if xgb is not None:
                    self.xgb_model = joblib.load(xgb_path)
                    print("XGBoost 로드")
                else:
                    print("XGBoost 모델 파일은 있으나 라이브러리 로드 불가")
            if os.path.exists(scaler_path):
                self.rf_scaler = joblib.load(scaler_path)
                print("Scaler 로드")
            if self.rf_model is None and self.xgb_model is None:
                print("AI 모델이 없어 통계 기반 예측 사용")
        except Exception as e:
            print(f"모델 로드 실패: {e}")

    def create_features_for_prediction(self, df):
        df = df.copy()
        df = df.sort_values('day')
        if 'weight_kg' in df.columns:
            df['weight'] = df['weight_kg']
        if 'feed_intake_kg' in df.columns:
            df['feed'] = df['feed_intake_kg']
        df['weight_lag1'] = df['weight'].shift(1)
        df['weight_lag3'] = df['weight'].shift(3)
        df['weight_lag7'] = df['weight'].shift(7)
        df['weight_rolling_mean_7'] = df['weight'].rolling(window=7, min_periods=1).mean()
        df['weight_rolling_std_7'] = df['weight'].rolling(window=7, min_periods=1).std()
        df['weight_change_1d'] = df['weight'] - df['weight_lag1']
        df['weight_change_3d'] = df['weight'] - df['weight_lag3']
        df['weight_change_7d'] = df['weight'] - df['weight_lag7']
        df['feed_weight_ratio'] = df['feed'] / df['weight']
        df['day_squared'] = df['day'] ** 2
        df['weight_squared'] = df['weight'] ** 2
        return df

    def predict_daily_gain(self, pig_data):
        df_features = self.create_features_for_prediction(pig_data)
        df_features = df_features.dropna()
        if len(df_features) == 0:
            return 0.65
        last_row = df_features.iloc[-1:]
        feature_cols = [
            'weight', 'day', 'feed',
            'weight_lag1', 'weight_lag3', 'weight_lag7',
            'weight_rolling_mean_7', 'weight_rolling_std_7',
            'weight_change_1d', 'weight_change_3d', 'weight_change_7d',
            'feed_weight_ratio', 'day_squared', 'weight_squared'
        ]
        X = last_row[feature_cols]
        predictions = []
        if self.rf_model is not None and self.rf_scaler is not None:
            X_scaled = self.rf_scaler.transform(X)
            rf_pred = self.rf_model.predict(X_scaled)[0]
            predictions.append(rf_pred)
        if self.xgb_model is not None:
            xgb_pred = self.xgb_model.predict(X)[0]
            predictions.append(xgb_pred)
        if len(predictions) > 0:
            return np.mean(predictions)
        else:
            return 0.65


class HybridPredictor:
    def __init__(self, target_weight=85):
        self.target_weight = target_weight
        self.ai_predictor = AIPredictor()
        self.lstm_predictor = LSTMPredictor()

    def train_lstm_on_data(self, df_data):
        print("\nLSTM 학습 데이터 준비 중...")
        all_pig_data = []
        for pig_id in df_data['pig_id'].unique():
            pig_data = df_data[df_data['pig_id'] == pig_id].sort_values('day')
            all_pig_data.append(pig_data)
        combined_data = pd.concat(all_pig_data, ignore_index=True)
        self.lstm_predictor.train(combined_data)

    def predict_shipment(self, df_data):
        print("\n출하 시점 예측 중...")
        self.train_lstm_on_data(df_data)
        results = []
        for pig_id in df_data['pig_id'].unique():
            pig_data = df_data[df_data['pig_id'] == pig_id].sort_values('day')
            current_weight = pig_data['weight_kg'].iloc[-1]
            current_day = pig_data['day'].iloc[-1]
            start_weight = pig_data['weight_kg'].iloc[0]
            if current_weight >= self.target_weight:
                results.append({
                    'pig_id': pig_id,
                    'current_weight': current_weight,
                    'days_to_shipment': 0,
                    'prediction_method': 'already_ready',
                    'status': 'ready'
                })
                continue
            ai_daily_gain = self.ai_predictor.predict_daily_gain(pig_data)
            remaining_weight = self.target_weight - current_weight
            ai_days = max(1, int(np.ceil(remaining_weight / ai_daily_gain)))
            lstm_predictions = self.lstm_predictor.predict_future_weights(pig_data, n_days=60)
            lstm_days = None
            if lstm_predictions is not None:
                for day, pred_weight in enumerate(lstm_predictions, 1):
                    if pred_weight >= self.target_weight:
                        lstm_days = day
                        break
                if lstm_days is None:
                    lstm_days = 60
            if len(pig_data) >= 7:
                recent_gain = pig_data['daily_gain_kg'].tail(7).mean()
            else:
                recent_gain = pig_data['daily_gain_kg'].mean()

            if recent_gain is None or pd.isna(recent_gain) or recent_gain <= 0.01:
                recent_gain = 0.6
            stat_days = max(1, int(np.ceil(remaining_weight / recent_gain)))
            predictions = []
            weights = []

            predictions.append(ai_days)
            weights.append(0.4)
            if lstm_days is not None:
                predictions.append(lstm_days)
                weights.append(0.4)
            else:
                weights[0] += 0.2
            predictions.append(stat_days)
            weights.append(0.2)
            weights = np.array(weights) / sum(weights)
            final_days = int(np.round(np.average(predictions, weights=weights)))
            min_days = max(1, int(remaining_weight / 1.2))
            max_days = int(remaining_weight / 0.3)
            final_days_clipped = np.clip(final_days, min_days, max_days)
            final_days_int = int(final_days_clipped)
            results.append({
                'pig_id': pig_id,
                'current_day': current_day,
                'current_weight': round(current_weight, 2),
                'start_weight': round(start_weight, 2),
                'remaining_weight': round(remaining_weight, 2),
                'ai_prediction_days': ai_days,
                'lstm_prediction_days': lstm_days if lstm_days else 'N/A',
                'stat_prediction_days': stat_days,
                'final_days_to_shipment': final_days_int,
                'predicted_shipment_date': (datetime.now() + timedelta(days=final_days_int)).strftime('%Y-%m-%d'),
                'prediction_method': 'hybrid_ensemble',
                'ai_daily_gain': round(ai_daily_gain, 3),
                'recent_daily_gain': round(recent_gain, 3),
                'status': 'predicted'
            })
        result_df = pd.DataFrame(results)
        print(f"\n예측 결과 요약:")
        print(f"- 출하 준비 완료: {len(result_df[result_df['status'] == 'ready'])}마리")
        print(f"- 출하 예정: {len(result_df[result_df['status'] == 'predicted'])}마리")
        predicted = result_df[result_df['status'] == 'predicted']
        if len(predicted) > 0:
            print(f"\n출하 예측:")
            print(f"- 평균 남은 기간: {predicted['final_days_to_shipment'].mean():.0f}일")
            print(f"- 최단 출하: {predicted['final_days_to_shipment'].min():.0f}일 후")
            print(f"- 최장 출하: {predicted['final_days_to_shipment'].max():.0f}일 후")
        return result_df

    def visualize_predictions(self, df_data, df_results, output_path='./step2_ai_predictions.png'):
        print("\n시각화 생성 중...")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        ax1 = axes[0, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(df_results)))
        for idx, (_, pig_info) in enumerate(df_results.iterrows()):
            pig_id = pig_info['pig_id']
            pig_data = df_data[df_data['pig_id'] == pig_id].sort_values('day')
            ax1.plot(pig_data['day'], pig_data['weight_kg'], label=f'Pig {pig_id}', color=colors[idx], linewidth=1.5)
        ax1.axhline(y=self.target_weight, color='red', linestyle='--', label=f'Target ({self.target_weight}kg)', linewidth=2)
        ax1.set_xlabel('Day'); ax1.set_ylabel('Weight (kg)'); ax1.set_title('Growth Curves', fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8); ax1.grid(True, alpha=0.3)
        ax2 = axes[0, 1]
        predicted = df_results[df_results['status'] == 'predicted']
        if len(predicted) > 0:
            x = np.arange(len(predicted)); width = 0.25
            if 'ai_prediction_days' in predicted.columns:
                ax2.bar(x - width, predicted['ai_prediction_days'], width, label='AI', alpha=0.7)
            if 'lstm_prediction_days' in predicted.columns:
                lstm_days = predicted['lstm_prediction_days'].replace('N/A', np.nan).astype(float)
                ax2.bar(x, lstm_days, width, label='LSTM', alpha=0.7)
            if 'stat_prediction_days' in predicted.columns:
                ax2.bar(x + width, predicted['stat_prediction_days'], width, label='Statistical', alpha=0.7)
            ax2.set_xticks(x)
            ax2.set_xticklabels([f'Pig {pid}' for pid in predicted['pig_id']], rotation=45)
            ax2.set_ylabel('Days to Shipment'); ax2.set_title('Prediction Method Comparison', fontweight='bold')
            ax2.legend(); ax2.grid(True, alpha=0.3, axis='y')
        ax3 = axes[0, 2]
        if 'final_days_to_shipment' in df_results.columns:
            valid_days = df_results[df_results['status'] == 'predicted']['final_days_to_shipment']
            if len(valid_days) > 0:
                ax3.hist(valid_days, bins=15, alpha=0.7, edgecolor='black')
                ax3.axvline(x=valid_days.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {valid_days.mean():.0f}d')
                ax3.set_xlabel('Days to Shipment'); ax3.set_ylabel('Frequency'); ax3.set_title('Final Prediction Distribution', fontweight='bold')
                ax3.legend(); ax3.grid(True, alpha=0.3, axis='y')
        ax4 = axes[1, 0]
        if 'ai_daily_gain' in df_results.columns:
            ax4.scatter(df_results['current_weight'], df_results['ai_daily_gain'], c=df_results.index, cmap='viridis', s=100, alpha=0.6)
            ax4.set_xlabel('Current Weight (kg)'); ax4.set_ylabel('AI Predicted Daily Gain (kg/day)'); ax4.set_title('Weight vs AI Predicted Growth Rate', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        ax5 = axes[1, 1]
        y_pos = np.arange(len(df_results))
        days = df_results['final_days_to_shipment'].fillna(0)
        colors_bar = ['green' if d == 0 else 'orange' if d < 30 else 'red' for d in days]
        bars = ax5.barh(y_pos, days, color=colors_bar, alpha=0.7)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels([f'Pig {pid}' for pid in df_results['pig_id']])
        ax5.set_xlabel('Days to Shipment'); ax5.set_title('Shipment Schedule', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        for bar, d in zip(bars, days):
            if d > 0:
                ax5.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{int(d)}d', va='center', fontsize=9)
        ax6 = axes[1, 2]
        methods = ['AI', 'LSTM', 'Statistical', 'Ensemble']
        if len(predicted) > 0:
            avg_ai = predicted['ai_prediction_days'].mean() if 'ai_prediction_days' in predicted.columns else 0
            avg_lstm = predicted['lstm_prediction_days'].replace('N/A', np.nan).astype(float).mean()
            avg_lstm = avg_lstm if not np.isnan(avg_lstm) else 0
            avg_stat = predicted['stat_prediction_days'].mean() if 'stat_prediction_days' in predicted.columns else 0
            avg_ensemble = predicted['final_days_to_shipment'].mean()
            values = [avg_ai, avg_lstm, avg_stat, avg_ensemble]
            bars = ax6.bar(methods, values, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
            ax6.set_ylabel('Average Days to Shipment'); ax6.set_title('Average Prediction by Method', fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='y')
            for bar in bars:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}d', ha='center', va='bottom')
        plt.suptitle('AI/LSTM Hybrid Prediction Results', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"그래프 저장: {output_path}")
        plt.close()

# -----------------------------------------------------------------
# (★신규★) 예측기 로드 함수
# -----------------------------------------------------------------
@st.cache_resource
def load_hybrid_predictor():
    """ AI/LSTM/통계 하이브리드 예측기를 로드합니다. """
    try:
        # (AI 담당자 코드의 target_weight=116 사용)
        predictor = HybridPredictor(target_weight=116.0)
        st.success("AI 하이브리드 예측기 로드 성공!")
        return predictor
    except Exception as e:
        st.error(f"하이브리드 예측기 로드 실패: {e}")
        return None

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

# -----------------------------------------------------------------
# (챔버 1, 2 돼지 20마리 샘플링 로직)
# -----------------------------------------------------------------
CHAMBER_IDS_TO_SAMPLE = [1, 2]  # (Chamber_no가 아닌 chamber_id 기준)
PIGS_PER_CHAMBER = 20
# 샘플링할 챔버 ID와 마리 수
if not pigs_df.empty and not pig_log_df_all.empty:

    try:
        # 1. '랜덤 시드'가 42로 고정된 '랜덤 생성기'를 만듭니다.
        rng = np.random.default_rng(42)

        # 2. 샘플링할 챔버의 돼지 ID 목록
        pigs_to_sample_list = []
        for cid in CHAMBER_IDS_TO_SAMPLE:
            pigs_in_chamber = pigs_df[pigs_df['chamber_id'] == cid]['pig_id'].unique()

            # 3. 챔버별 20마리 샘플링 (20마리보다 적으면 모두 선택)
            sample_size = min(len(pigs_in_chamber), PIGS_PER_CHAMBER)

            # 4. 'np.random.choice' 대신, 시드가 고정된 'rng.choice'를 사용합니다.
            sampled_pig_ids = rng.choice(pigs_in_chamber, size=sample_size, replace=False)
            pigs_to_sample_list.append(sampled_pig_ids)

        # 5. 샘플링하지 않을 챔버(3, 4번)의 돼지 ID 목록
        pigs_to_keep_ids = pigs_df[~pigs_df['chamber_id'].isin(CHAMBER_IDS_TO_SAMPLE)]['pig_id'].unique()
        pigs_to_sample_list.append(pigs_to_keep_ids)

        # 6. 최종 사용할 돼지 ID 목록
        final_pig_ids = np.concatenate(pigs_to_sample_list)

        # 7. 'Pigs' (마스터)와 'Pig_Logs' (로그) 테이블 모두를 이 ID 목록으로 필터링
        pigs_df = pigs_df[pigs_df['pig_id'].isin(final_pig_ids)].copy()
        pig_log_df_all = pig_log_df_all[pig_log_df_all['pig_id'].isin(final_pig_ids)].copy()

    except Exception as e:
        st.error(f"샘플링 중 오류: {e}")

# =================================================================
# A. '전체 맵 (Overview)' 화면
# =================================================================
if st.session_state.view_mode == "overview":

    st.title("🐷 스마트 축사 현황 (전체 맵)")

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

                #  AI 규칙 엔진 적용
                # 1. 각 돼지의 정상 체온 범위를 계산
                latest_pig_logs['tmin'], latest_pig_logs['tmax'] = zip(
                    *latest_pig_logs['weight_kg'].apply(get_normal_temp_range))

                # 2. (호흡 기준은 55~70으로 가정 - 필요시 수정)
                breath_norm_min = 55
                breath_norm_max = 70

                # '정상' 범위를 벗어나는 모든 개체 필터링
                warning_pigs_total = latest_pig_logs[
                    (latest_pig_logs["temp_rectal"] < latest_pig_logs["tmin"]) |
                    (latest_pig_logs["temp_rectal"] > latest_pig_logs["tmax"]) |
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

        # (AI 규칙을 적용하기 위해 'pig_log_df_all'의 유효한 최신 로그를 미리 계산)
        valid_logs_all = pd.DataFrame()
        if not pig_log_df_all.empty:
            valid_logs_all = pig_log_df_all.dropna(subset=['temp_rectal', 'breath_rate', 'weight_kg'])
            if not valid_logs_all.empty:
                valid_logs_all = valid_logs_all.loc[valid_logs_all.groupby("pig_id")["timestamp"].idxmax()]
                valid_logs_all['tmin'], valid_logs_all['tmax'] = zip(
                    *valid_logs_all['weight_kg'].apply(get_normal_temp_range))

        for i, row in chambers_df.iterrows():
            chamber_id = row['chamber_id']
            chamber_no = row['chamber_no']
            current_col = grid_cols[i % 2]

            warn_count = 0
            if not pigs_df.empty and not valid_logs_all.empty:
                pigs_in_chamber_ids = pigs_df[pigs_df['chamber_id'] == chamber_id]['pig_id']

                latest_logs_chamber = valid_logs_all[valid_logs_all['pig_id'].isin(pigs_in_chamber_ids)]

                if not latest_logs_chamber.empty:
                    # (호흡 기준은 55~70으로 가정 - 필요시 수정)
                    breath_norm_min = 55
                    breath_norm_max = 70

                    warning_pigs_chamber = latest_logs_chamber[
                        (latest_logs_chamber["temp_rectal"] < latest_logs_chamber["tmin"]) |
                        (latest_logs_chamber["temp_rectal"] > latest_logs_chamber["tmax"]) |
                        (latest_logs_chamber["breath_rate"] < breath_norm_min) |
                        (latest_logs_chamber["breath_rate"] > breath_norm_max)
                        ]
                    warn_count = len(warning_pigs_chamber)

            with current_col.container(border=True):

                #'주의' 배너 기준 설정 (5마리 이상)
                warning_threshold_count = 5

                if warn_count >= warning_threshold_count:
                    st.error(f"🚨 {chamber_no}번 챔버 (주의!)")
                else:
                    st.subheader(f" {chamber_no}번 챔버")

                c1_metric, c2_metric = st.columns(2)
                # (현재 온도 로직)
                chamber_sensor_data = sensor_df_all[sensor_df_all['chamber_id'] == chamber_id]
                if not chamber_sensor_data.empty and "temperature" in chamber_sensor_data.columns:
                    c1_metric.metric("현재 온도", f"{chamber_sensor_data.iloc[0]['temperature']:.1f} °C")
                else:
                    c1_metric.metric("현재 온도", "N/A")

                c2_metric.metric("건강 '주의' 개체", f"{warn_count} 마리")

                st.button(
                    f"{chamber_no}번 챔버 상세 정보 보기",
                    key=f"btn_detail_{chamber_id}",
                    on_click=set_detail_view,
                    args=(chamber_id, chamber_no)
                )
    # 일일 날씨 (시간별 상세 예보 DB)
    st.header("🌦️ 일일 날씨")

    # (대문자로 변환된 컬럼명 사용)
    needed_weather_cols = {"FCST_DT", "T1H", "REH", "RN1", "SKY", "PTY"}

    if not weather_ultra_fcst_df.empty and needed_weather_cols.issubset(weather_ultra_fcst_df.columns):

        weather_chart_data = weather_ultra_fcst_df.set_index("FCST_DT")

        w_tab1, w_tab2, w_tab3 = st.tabs(["🌡️ 기온 (T1H)", "💧 습도 (REH)", "☔ 시간당 강수량 (RN1)"])

        with w_tab1:
            st.plotly_chart(px.line(weather_chart_data, y='T1H', title='시간별 외부 기온'), width='stretch')
        with w_tab2:
            st.plotly_chart(px.line(weather_chart_data, y='REH', title='시간별 외부 습도'), width='stretch')
        with w_tab3:
            st.plotly_chart(px.bar(weather_chart_data, y='RN1', title='시간별 강수량'), width='stretch')

        latest_sky = weather_ultra_fcst_df.iloc[0].get("SKY", -1)
        st.info(f"현재 하늘 상태(SKY) 코드는 '{latest_sky}'입니다. (1: 맑음, 3: 구름많음, 4: 흐림)")

    else:
        st.warning("시간별 상세 날씨(weather_ultra_fcst) 데이터를 DB에서 불러오지 못했거나, 필요한 컬럼이 없습니다.")

    st.divider()
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
            width='stretch'
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
        pig_log_df_filtered = pig_log_df_all[pig_log_df_all['pig_id'].isin(pigs_in_chamber)].copy()

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

            # 1. 유효한 건강 데이터 필터링
            valid_health_logs = pig_log_df_filtered.dropna(subset=['temp_rectal', 'breath_rate', 'weight_kg'])

            if not valid_health_logs.empty:
                # 2. 각 돼지의 가장 최신 로그 가져오기
                latest_pig_logs = valid_health_logs.loc[valid_health_logs.groupby('pig_id')['timestamp'].idxmax()]

                # 3. AI 규칙 엔진 적용
                latest_pig_logs['tmin'], latest_pig_logs['tmax'] = zip(
                    *latest_pig_logs['weight_kg'].apply(get_normal_temp_range))

                # 4. (호흡 기준은 55~70으로 가정 - 필요시 수정)
                breath_norm_min = 55
                breath_norm_max = 70

                # 5. '정상' 범위를 벗어나는 모든 개체 필터링
                warning_pigs = latest_pig_logs[
                    (latest_pig_logs['temp_rectal'] < latest_pig_logs["tmin"]) |  # 온도 낮음
                    (latest_pig_logs['temp_rectal'] > latest_pig_logs["tmax"]) |  # 온도 높음
                    (latest_pig_logs['breath_rate'] < breath_norm_min) |  # 호흡 느림
                    (latest_pig_logs['breath_rate'] > breath_norm_max)  # 호흡 빠름
                    ]

                st.metric("건강 '주의' 개체 수", f"{len(warning_pigs)} 마리")

                if len(warning_pigs) > 0:
                    with st.expander("'주의' 개체 목록 보기"):

                        # 6. '주의 원인'을 AI 규칙 기준으로 변경
                        def find_reason(row):
                            reasons = []
                            tmin, tmax = row['tmin'], row['tmax']  # (AI가 계산한 범위)
                            # 온도 확인
                            if row['temp_rectal'] < tmin:
                                reasons.append(f"온도 낮음 ({row['temp_rectal']:.1f}°C)")
                            elif row['temp_rectal'] > tmax:
                                reasons.append(f"온도 높음 ({row['temp_rectal']:.1f}°C)")

                            # 호흡 확인
                            if row['breath_rate'] < breath_norm_min:
                                reasons.append(f"호흡 느림 ({row['breath_rate']:.0f}회)")
                            elif row['breath_rate'] > breath_norm_max:
                                reasons.append(f"호흡 빠름 ({row['breath_rate']:.0f}회)")

                            return ', '.join(reasons)


                        warning_pigs_with_reason = warning_pigs.copy()
                        warning_pigs_with_reason['주의 원인'] = warning_pigs_with_reason.apply(find_reason, axis=1)

                        display_cols = ["pig_id", "weight_kg", "temp_rectal", "breath_rate", "주의 원인"]
                        st.dataframe(warning_pigs_with_reason[display_cols])
            else:
                st.warning("유효한 건강 데이터(체온/호흡수/체중)가 없습니다.")
        else:
            st.warning("돼지 로그 데이터를 찾을 수 없습니다.")

    st.divider()

    # --- 섹션 3: 출하 및 에너지 분석 ---
    st.header("🐖 출하 및 에너지 분석")
    tab1, tab2 = st.tabs(["출하 날짜 예측", "에너지 사용량 분석"])

    with tab1:
        # 1. 앱 시작 시 로드한 '하이브리드 예측기'를 가져옵니다.
        predictor = load_hybrid_predictor()

        target_weight = st.number_input(
            "목표 출하 체중(kg)을 입력하세요:",
            min_value=80.0, value=116.0, step=1.0,
            help="이 체중을 기준으로 출하 가능 개체 수와 예측 날짜를 계산합니다."
        )
        if predictor is not None:
            predictor.target_weight = target_weight

        if not pig_log_df_filtered.empty and predictor is not None:

            # (데이터 병합 및 AI 입력용 데이터 생성)
            feed_data_df = sensor_df_filtered[['timestamp', 'feed_volume']].dropna()
            pig_data_merged = pd.merge(
                pig_log_df_filtered,
                feed_data_df,
                on="timestamp",
                how="left"
            )
            pig_data_for_ai = pig_data_merged.rename(columns={
                'weight_kg': 'weight_kg',
                'feed_volume': 'feed_intake_kg',
                'pig_id': 'pig_id'
            })
            if 'day' not in pig_data_for_ai.columns:
                pig_data_for_ai = pig_data_for_ai.sort_values(by=['pig_id', 'timestamp'])
                pig_data_for_ai['day'] = pig_data_for_ai.groupby('pig_id')['timestamp'].transform(
                    lambda x: (x - x.min()).dt.days)
            if 'daily_gain_kg' not in pig_data_for_ai.columns:
                pig_data_for_ai['weight_lag1'] = pig_data_for_ai.groupby('pig_id')['weight_kg'].shift(1)
                pig_data_for_ai['daily_gain_kg'] = pig_data_for_ai['weight_kg'] - pig_data_for_ai['weight_lag1']
                pig_data_for_ai['daily_gain_kg'] = pig_data_for_ai['daily_gain_kg'].fillna(0.6)

            # ----------------------------------------------------

            logs_with_weights = (
                pig_data_for_ai.dropna(subset=["weight_kg"])
                if "weight_kg" in pig_data_for_ai.columns else pd.DataFrame()
            )

            if not logs_with_weights.empty:
                latest_weights = logs_with_weights.loc[
                    logs_with_weights.groupby("pig_id")["timestamp"].idxmax()
                ]
                ship_ready_now = latest_weights[latest_weights["weight_kg"] >= target_weight]

                c1, c2 = st.columns(2)
                c1.metric(f"현재 {target_weight}kg 이상 (출하 가능)", f"{len(ship_ready_now)} 마리")
                pigs_below = latest_weights[latest_weights["weight_kg"] < target_weight]
                c2.metric("출하 예측 대상", f"{len(pigs_below)} 마리")
                st.divider()

                st.subheader(f"🐷 {target_weight}kg 도달 날짜 예측 (AI 하이브리드)")

                if not pigs_below.empty:

                    results = []
                    today = pd.Timestamp.now()

                    with st.spinner(f"{len(pigs_below)}마리 전체에 대한 출하 예측을 계산 중입니다... (시간 소요)"):
                        # (LSTM 모델은 모든 돼지 데이터로 1회 훈련 필요)
                        predictor.train_lstm_on_data(logs_with_weights)

                        for _, rep_pig in pigs_below.iterrows():
                            pig_id = rep_pig["pig_id"]
                            current_weight = rep_pig["weight_kg"]

                            pig_data_hist = logs_with_weights[logs_with_weights['pig_id'] == pig_id]

                            prediction_result_df = predictor.predict_shipment(pig_data_hist)

                            if not prediction_result_df.empty:
                                pred_row = prediction_result_df.iloc[0]

                                # 1. 'results' 리스트에 4개의 핵심 정보만 저장합니다.
                                results.append({
                                    '돼지 ID': pig_id,
                                    '현재 체중(kg)': round(current_weight, 1),
                                    '남은 일수(일)': int(pred_row['final_days_to_shipment']),
                                    '예상 출하 날짜': pred_row['predicted_shipment_date']
                                })

                    if results:
                        # 6. 예측 결과 테이블(DataFrame) 생성
                        result_df = pd.DataFrame(results).sort_values('남은 일수(일)')

                        fastest_pig = result_df.iloc[0]
                        st.metric(
                            f"가장 빠른 예상 출하일 (ID: {fastest_pig['돼지 ID']})",
                            f"{fastest_pig['예상 출하 날짜']}",
                            f"{fastest_pig['남은 일수(일)']}일 남음"
                        )

                        with st.expander("전체 개체별 예상 출하일 보기 (빠른 순)"):
                            #2. 'result_df' (4개 컬럼만 있음)를 인덱스 설정 후 바로 표시합니다.
                            st.dataframe(result_df.set_index('돼지 ID'), width='stretch')
                    else:
                        st.error("AI 예측 중 오류가 발생했습니다.")

                else:
                    st.success(f"데이터가 있는 모든 개체가 이미 목표 체중({target_weight}kg) 이상입니다.")
            else:
                st.warning("이 챔버에는 현재 유효한 체중 데이터가 없습니다.")
        else:
            st.warning("몸무게 데이터가 없거나 AI 예측기를 로드하지 못했습니다.")
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
