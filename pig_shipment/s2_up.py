import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
import os
import warnings
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
try:
    import xgboost as xgb
except ImportError:
    xgb = None
    print("XGBoost 라이브러리가 로드되지 않았습니다. XGBoost 예측은 건너뛰어집니다.")

warnings.filterwarnings('ignore')

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except:
        pass

target_weight = 116

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


def main():
    print("=" * 80)
    print("돼지 출하 예측 시스템 - Step 2: AI/LSTM 예측")
    print("=" * 80)
    generator = DummyDataGenerator()
    dummy_data = generator.generate_dummy_dataset(n_pigs=10, n_days=60)
    dummy_data.to_csv('./step2_dummy_data_ai.csv', index=False, encoding='utf-8-sig')
    print(f"더미 데이터 저장: ./step2_dummy_data_ai.csv")
    predictor = HybridPredictor(target_weight=target_weight)
    results = predictor.predict_shipment(dummy_data)
    results.to_csv('./step2_ai_prediction_results.csv', index=False, encoding='utf-8-sig')
    print(f"예측 결과 저장: ./step2_ai_prediction_results.csv")
    predictor.visualize_predictions(dummy_data, results)
    print("\n" + "=" * 80)
    print("Step 2 완료!")
    print("=" * 80)
    print("\n생성된 파일:")
    print("1. step2_dummy_data_ai.csv")
    print("2. step2_ai_prediction_results.csv")
    print("3. step2_ai_predictions.png")
    print("\n" + "=" * 80)
    print("상세 예측 결과")
    print("=" * 80)
    for _, pig in results.iterrows():
        print(f"\nPig {pig['pig_id']}:")
        print(f"현재 체중: {pig['current_weight']:.1f}kg")
        if pig['status'] == 'ready':
            print("출하 준비 완료")
        else:
            print(f"AI 예측: {pig['ai_prediction_days']:.0f}일")
            print(f"LSTM 예측: {pig['lstm_prediction_days']}")
            print(f"통계 예측: {pig['stat_prediction_days']:.0f}일")
            print(f"최종 예측: {pig['final_days_to_shipment']:.0f}일 후")
            print(f"예상 출하날짜: {pig['predicted_shipment_date']}")

if __name__ == "__main__":
    main()
