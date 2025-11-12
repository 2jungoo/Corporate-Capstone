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
import matplotlib.pyplot as plt

target_weight = 80.0 # (대시보드 기준 80kg로 수정)

class DummyDataGenerator:
    """학습된 패턴 기반 2달치 더미 데이터 생성"""

    def __init__(self, pattern_file='./growth_patterns.json'):
        self.pattern_file = pattern_file
        self.target_weight = target_weight
        self.patterns = None
        self.load_patterns()

    def load_patterns(self):
        """학습된 패턴 로드"""
        print("\n" + "=" * 80)
        print("📚 학습된 패턴 로딩 중...")
        print("=" * 80)

        if not os.path.exists(self.pattern_file):
            # 패턴 파일이 없으면 기본값 사용
            print("⚠️ 패턴 파일이 없어 기본값 사용")
            self.patterns = self.get_default_patterns()
        else:
            with open(self.pattern_file, 'r', encoding='utf-8') as f:
                self.patterns = json.load(f)

        print("✓ 패턴 로드 완료")
        print(f"\n📊 로드된 정보:")
        print(f"   - 전체 평균 증체율: {self.patterns['overall']['mean_daily_gain']:.3f} kg/일")
        print(f"   - 체중 범위: {self.patterns['overall']['min_weight']:.1f}~{self.patterns['overall']['max_weight']:.1f} kg")

        return self.patterns

    def get_default_patterns(self):
        """기본 성장 패턴 (업계 표준 기반)"""
        return {
            'overall': {
                'mean_daily_gain': 0.771,  # 실제로는 0.6-0.8 정도가 정상
                'std_daily_gain': 0.15,
                'min_weight': 20.0,
                'max_weight': 110.0,
                'mean_weight': 50.0
            },
            'weight_bins': {
                '0-20kg': {
                    'mean_daily_gain': 0.45,  # 어린 돼지는 느림
                    'std_daily_gain': 0.10,
                    'mean_feed': 0.8
                },
                '20-40kg': {
                    'mean_daily_gain': 0.65,  # 점점 빨라짐
                    'std_daily_gain': 0.12,
                    'mean_feed': 1.5
                },
                '40-60kg': {
                    'mean_daily_gain': 0.75,  # 최적 성장기
                    'std_daily_gain': 0.10,
                    'mean_feed': 2.2
                },
                '60-80kg': {
                    'mean_daily_gain': 0.80,  # 여전히 빠름
                    'std_daily_gain': 0.10,
                    'mean_feed': 2.8
                },
                '80-100kg': {
                    'mean_daily_gain': 0.70,  # 점점 느려짐
                    'std_daily_gain': 0.12,
                    'mean_feed': 3.2
                },
                '100kg+': {
                    'mean_daily_gain': 0.55,  # 비육 후기
                    'std_daily_gain': 0.15,
                    'mean_feed': 3.5
                }
            }
        }

    def get_daily_gain_for_weight(self, weight, day):
        """
        특정 체중과 일령에 대한 일일 증체량 추정
        더 현실적인 성장 곡선 적용
        """
        weight_bins = {
            '0-20kg': (0, 20),
            '20-40kg': (20, 40),
            '40-60kg': (40, 60),
            '60-80kg': (60, 80),
            '80-100kg': (80, 100),
            '100kg+': (100, 200)
        }

        # 해당 체중 구간 찾기
        for bin_name, (min_w, max_w) in weight_bins.items():
            if min_w <= weight < max_w:
                if bin_name in self.patterns['weight_bins']:
                    bin_data = self.patterns['weight_bins'][bin_name]

                    # 평균과 표준편차
                    mean_gain = bin_data['mean_daily_gain']
                    std_gain = bin_data.get('std_daily_gain', mean_gain * 0.15)

                    # 정규분포에서 샘플링 (변동성 줄임)
                    daily_gain = np.random.normal(mean_gain, std_gain * 0.3)

                    # 성장 곡선 효과 (S자 곡선)
                    # 초기와 후기에는 느리고 중간에 빠름
                    growth_factor = 1.0
                    if day < 30:  # 초기 적응기
                        growth_factor = 0.7 + (day / 30) * 0.3
                    elif day > 120:  # 비육 후기
                        growth_factor = max(0.6, 1.0 - (day - 120) / 200)

                    daily_gain *= growth_factor

                    # 범위 제한 (최소 0.2kg, 최대 1.2kg)
                    daily_gain = np.clip(daily_gain, 0.2, 1.2)

                    return daily_gain

        # 기본값
        return 0.6

    def get_feed_for_weight(self, weight):
        """
        특정 체중에 대한 사료 섭취량 추정
        일반적으로 체중의 3-4%
        """
        # 체중별 사료 섭취 비율
        if weight < 30:
            feed_ratio = 0.04  # 4%
        elif weight < 60:
            feed_ratio = 0.035  # 3.5%
        elif weight < 90:
            feed_ratio = 0.03  # 3%
        else:
            feed_ratio = 0.025  # 2.5%

        base_feed = weight * feed_ratio

        # 약간의 일별 변동
        feed = base_feed * np.random.uniform(0.95, 1.05)

        return max(0.5, feed)  # 최소 0.5kg

    def generate_pig_data(self, pig_id, start_weight=None, n_days=60):
        """
        한 마리 돼지의 n일치 데이터 생성 (정상적인 성장)
        """
        # 시작 체중 (20-30kg 범위의 이유자돈)
        if start_weight is None:
            start_weight = np.random.uniform(20, 30)

        data = []
        current_weight = start_weight

        for day in range(n_days):
            # 일일 증체량 (체중과 일령 고려)
            daily_gain = self.get_daily_gain_for_weight(current_weight, day)

            # 체중 업데이트 (확실히 증가)
            current_weight = current_weight + abs(daily_gain)  # 절대값으로 항상 증가

            # 측정 오차 (작게)
            noise = np.random.normal(0, 0.1)
            measured_weight = current_weight + noise

            # 사료 섭취량
            feed_intake = self.get_feed_for_weight(measured_weight)

            # 환경 데이터
            temperature = 22 + np.random.normal(0, 2)
            humidity = 65 + np.random.normal(0, 5)

            # 급수량 (체중의 8-10%)
            water_intake = measured_weight * np.random.uniform(0.08, 0.10)

            # 활동량 점수 (임의)
            activity_score = np.random.uniform(3, 8)

            # 건강 상태 (대부분 정상)
            health_status = np.random.choice(['good', 'normal', 'attention'], p=[0.7, 0.25, 0.05])

            data.append({
                'pig_id': pig_id,
                'day': day,
                'weight_kg': round(measured_weight, 2),
                'daily_gain_kg': round(daily_gain, 3),
                'feed_intake_kg': round(feed_intake, 2),
                'water_intake_l': round(water_intake, 2),
                'temperature_c': round(temperature, 1),
                'humidity_percent': round(humidity, 1),
                'activity_score': round(activity_score, 1),
                'health_status': health_status,
                'chamber': 'chamber_new',
                'date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d')
            })

        return pd.DataFrame(data)

    def generate_dummy_dataset(self, n_pigs=10, n_days=60, output_path='./step2_dummy_data.csv'):
        """
        여러 돼지의 2달치 더미 데이터 생성
        """
        print("\n" + "=" * 80)
        print("🎲 더미 데이터 생성 중...")
        print("=" * 80)

        all_data = []

        for i in range(n_pigs):
            # 각 돼지마다 약간 다른 시작 체중
            start_weight = np.random.uniform(18, 28)
            pig_data = self.generate_pig_data(pig_id=i+1, start_weight=start_weight, n_days=n_days)
            all_data.append(pig_data)

            # 진행상황 표시
            if (i+1) % 5 == 0:
                print(f"   {i+1}/{n_pigs} 마리 생성 완료...")

        df = pd.concat(all_data, ignore_index=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"\n✓ 데이터 생성 완료!")
        print(f"   - 돼지 수: {n_pigs}마리")
        print(f"   - 기간: {n_days}일 (약 2개월)")
        print(f"   - 총 데이터: {len(df)}건")
        print(f"   - 저장 위치: {output_path}")

        # 통계
        print(f"\n📊 생성된 데이터 통계:")
        start_weights = df[df['day'] == 0]['weight_kg']
        end_weights = df[df['day'] == n_days-1]['weight_kg']
        print(f"   - 평균 시작 체중: {start_weights.mean():.2f}kg")
        print(f"   - 평균 종료 체중: {end_weights.mean():.2f}kg")
        print(f"   - 평균 총 증체량: {(end_weights.mean() - start_weights.mean()):.2f}kg")
        print(f"   - 평균 일일 증체량: {(end_weights.mean() - start_weights.mean())/n_days:.3f}kg/일")

        return df


class PatternBasedPredictor:
    """학습된 패턴 기반 예측 (수정본)"""

    def __init__(self,target_weight ,pattern_file='./growth_patterns.json'):
        self.pattern_file = pattern_file
        #self.target_weight = target_weight
        self.patterns = None
        self.load_patterns()

    def load_patterns(self):
        """패턴 로드"""
        if os.path.exists(self.pattern_file):
            with open(self.pattern_file, 'r', encoding='utf-8') as f:
                self.patterns = json.load(f)
        else:
            # 기본 패턴 사용
            generator = DummyDataGenerator()
            self.patterns = generator.patterns
        return self.patterns

    def predict_shipment_days(self, current_weight, current_age, recent_daily_gain, target_weight):
        """
        출하까지 남은 일수 예측 (개선된 버전)
        """
        if current_weight >= target_weight:
            return 0

        remaining_weight = target_weight - current_weight

        # 최근 성장률이 있으면 사용
        if recent_daily_gain > 0.1:  # 최소 0.1kg/일 이상
            # 나이에 따른 성장률 감소 고려
            age_factor = 1.0
            if current_age > 120:
                age_factor = max(0.7, 1.0 - (current_age - 120) / 200)

            adjusted_gain = recent_daily_gain * age_factor

            # 체중에 따른 성장률 조정
            if current_weight > 60:
                adjusted_gain *= 0.9  # 무거워질수록 느려짐
            if current_weight > 90:
                adjusted_gain *= 0.8
        else:
            # 체중별 기본 성장률 사용
            if current_weight < 40:
                adjusted_gain = 0.65
            elif current_weight < 60:
                adjusted_gain = 0.75
            elif current_weight < 80:
                adjusted_gain = 0.70
            else:
                adjusted_gain = 0.60

        if adjusted_gain <= 0:
            adjusted_gain = 0.5  # 최소값

        days_to_shipment = remaining_weight / adjusted_gain

        # 최대 180일로 제한 (6개월)
        return min(180, int(np.ceil(days_to_shipment)))

    def analyze_new_pigs(self, data_path):
        """신규 돼지 데이터 분석 및 예측"""
        print("\n" + "=" * 80)
        print("📊 신규 돼지 분석 중...")
        print("=" * 80)

        df = pd.read_csv(data_path)

        print(f"✓ 데이터 로드: {len(df)}건")
        print(f"   - 돼지 수: {df['pig_id'].nunique()}마리")
        print(f"   - 기간: {df['day'].max() + 1}일")

        results = []

        for pig_id in df['pig_id'].unique():
            pig_data = df[df['pig_id'] == pig_id].sort_values('day')

            # 현재 상태
            last_record = pig_data.iloc[-1]
            first_record = pig_data.iloc[0]

            current_day = int(last_record['day'])
            current_weight = last_record['weight_kg']
            start_weight = first_record['weight_kg']

            # 최근 14일 증체율 계산 (더 긴 기간으로 안정적 계산)
            if len(pig_data) >= 14:
                recent = pig_data.tail(14)
            elif len(pig_data) >= 7:
                recent = pig_data.tail(7)
            else:
                recent = pig_data

            if len(recent) > 1:
                weight_diff = recent['weight_kg'].iloc[-1] - recent['weight_kg'].iloc[0]
                days_diff = recent['day'].iloc[-1] - recent['day'].iloc[0]
                if days_diff > 0:
                    recent_daily_gain = weight_diff / days_diff
                else:
                    recent_daily_gain = 0.6  # 기본값
            else:
                recent_daily_gain = 0.6  # 기본값

            # 음수 방지
            recent_daily_gain = max(0.2, recent_daily_gain)

            # 전체 기간 평균 증체율
            total_gain = current_weight - start_weight
            total_days = current_day + 1
            avg_daily_gain = total_gain / total_days if total_days > 0 else 0.6
            avg_daily_gain = max(0.2, avg_daily_gain)

            # 예상 나이 (시작을 30일령으로 가정)
            estimated_age = 30 + current_day

            # 출하 시점 예측
            days_to_shipment = self.predict_shipment_days(
                current_weight, estimated_age, recent_daily_gain
            )

            if days_to_shipment is not None:
                shipment_day = current_day + days_to_shipment
                status = 'ready' if days_to_shipment == 0 else 'predicted'
                predicted_date = (pd.to_datetime(last_record['date']) +
                                timedelta(days=days_to_shipment)).strftime('%Y-%m-%d')
            else:
                shipment_day = None
                days_to_shipment = None
                status = 'error'
                predicted_date = None

            results.append({
                'pig_id': pig_id,
                'chamber': last_record['chamber'],
                'current_day': current_day,
                'current_weight': round(current_weight, 2),
                'start_weight': round(start_weight, 2),
                'total_gain': round(total_gain, 2),
                'avg_daily_gain': round(avg_daily_gain, 3),
                'recent_daily_gain': round(recent_daily_gain, 3),
                'estimated_age': estimated_age,
                'remaining_weight': round(self.target_weight - current_weight, 2) if current_weight < self.target_weight else 0,
                'days_to_shipment': days_to_shipment,
                'total_days_to_market': estimated_age + days_to_shipment if days_to_shipment else None,
                'shipment_day': shipment_day,
                'predicted_shipment_date': predicted_date,
                'status': status
            })

        result_df = pd.DataFrame(results)

        # 요약
        print(f"\n📊 예측 결과 요약:")
        print(f"   - 출하 준비 완료: {len(result_df[result_df['status'] == 'ready'])}마리")
        print(f"   - 출하 예정: {len(result_df[result_df['status'] == 'predicted'])}마리")

        predicted = result_df[result_df['status'] == 'predicted']
        if len(predicted) > 0:
            print(f"\n   📅 출하 예측:")
            print(f"   - 평균 남은 기간: {predicted['days_to_shipment'].mean():.0f}일")
            print(f"   - 최단 출하: {predicted['days_to_shipment'].min():.0f}일 후")
            print(f"   - 최장 출하: {predicted['days_to_shipment'].max():.0f}일 후")
            print(f"   - 평균 출하 일령: {predicted['total_days_to_market'].mean():.0f}일령")

        return result_df

    def visualize_predictions(self, df_data, df_results, output_path='./step2_predictions_fixed.png'):
        """예측 결과 시각화"""
        print("\n📊 시각화 생성 중...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 성장 곡선과 예측
        ax1 = axes[0, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(df_results)))

        for idx, (_, pig_info) in enumerate(df_results.iterrows()):
            pig_id = pig_info['pig_id']
            pig_data = df_data[df_data['pig_id'] == pig_id].sort_values('day')

            # 실제 데이터
            ax1.plot(pig_data['day'], pig_data['weight_kg'],
                    label=f'Pig {pig_id}', color=colors[idx], linewidth=1.5)

            # 예측 (점선)
            if pig_info['days_to_shipment'] and pig_info['days_to_shipment'] > 0:
                future_days = np.arange(pig_data['day'].max(),
                                      pig_data['day'].max() + pig_info['days_to_shipment'] + 1)
                future_weights = np.linspace(pig_info['current_weight'],
                                           self.target_weight, len(future_days))
                ax1.plot(future_days, future_weights, '--',
                        color=colors[idx], alpha=0.5, linewidth=1)

        ax1.axhline(y=self.target_weight, color='red', linestyle='--',
                   label=f'Target ({self.target_weight}kg)', linewidth=2)
        ax1.set_xlabel('Day', fontsize=12)
        ax1.set_ylabel('Weight (kg)', fontsize=12)
        ax1.set_title('Growth Curves (2 Months)', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 2. 현재 체중 vs 증체율
        ax2 = axes[0, 1]
        ax2.scatter(df_results['current_weight'], df_results['recent_daily_gain'],
                   c=df_results.index, cmap='viridis', s=100, alpha=0.6)

        # 평균선
        ax2.axhline(y=df_results['recent_daily_gain'].mean(),
                   color='red', linestyle='--',
                   label=f"Pattern Avg: {df_results['recent_daily_gain'].mean():.3f}",
                   linewidth=2)

        ax2.set_xlabel('Current Weight (kg)', fontsize=12)
        ax2.set_ylabel('Recent Daily Gain (kg/day)', fontsize=12)
        ax2.set_title('Weight vs Growth Rate', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 출하 예정일 분포
        ax3 = axes[1, 0]
        if 'days_to_shipment' in df_results.columns:
            valid_days = df_results['days_to_shipment'].dropna()
            if len(valid_days) > 0:
                ax3.hist(valid_days, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax3.axvline(x=valid_days.mean(), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {valid_days.mean():.0f} days')
                ax3.set_xlabel('Days to Shipment', fontsize=12)
                ax3.set_ylabel('Frequency', fontsize=12)
                ax3.set_title('Distribution of Remaining Days', fontsize=14, fontweight='bold')
                ax3.legend()
                ax3.grid(True, alpha=0.3, axis='y')

        # 4. 개별 출하 스케줄
        ax4 = axes[1, 1]
        y_pos = np.arange(len(df_results))
        days_to_ship = df_results['days_to_shipment'].fillna(0)
        colors_bar = ['green' if d == 0 else 'orange' if d < 30 else 'red'
                     for d in days_to_ship]

        bars = ax4.barh(y_pos, days_to_ship, color=colors_bar, alpha=0.7)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([f'Pig {pid}' for pid in df_results['pig_id']])
        ax4.set_xlabel('Days to Shipment', fontsize=12)
        ax4.set_title('Individual Shipment Schedule', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # 막대 위에 일수 표시
        for bar, days in zip(bars, days_to_ship):
            if days > 0:
                ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{int(days)}d', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 그래프 저장: {output_path}")
        plt.close()

# -----------------------------------------------------------------
# (★신규★) 예측기 로드 함수
# -----------------------------------------------------------------
@st.cache_resource
def load_shipment_predictor():
    """ 'growth_patterns_ai.json'을 로드한 예측기를 생성합니다. """
    try:
        # (파일 경로를 우리 프로젝트에 맞게 수정)
        predictor = PatternBasedPredictor(target_weight=80.0, pattern_file='./growth_patterns_ai.json')
        st.success("AI 출하 예측기(통계) 로드 성공!")
        return predictor
    except Exception as e:
        st.error(f"출하 예측기 로드 실패: {e}")
        st.info("growth_patterns_ai.json 파일이 project.py와 같은 폴더에 있는지 확인하세요.")
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
                                reasons.append(f"온도 낮음")
                            elif row['temp_rectal'] > temp_norm_max:
                                reasons.append(f"온도 높음")

                            # 호흡 확인
                            if row['breath_rate'] < breath_norm_min:
                                reasons.append(f"호흡 느림")
                            elif row['breath_rate'] > breath_norm_max:
                                reasons.append(f"호흡 빠름")

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

    # --- 섹션 3: 출하 및 에너지 분석 ---
    st.header("🐖 출하 및 에너지 분석")
    tab1, tab2 = st.tabs(["출하 날짜 예측", "에너지 사용량 분석"])

    with tab1:
        # 1. 앱 시작 시 로드한 예측기를 가져옵니다.
        predictor = load_shipment_predictor()

        target_weight = st.number_input(
            "목표 출하 체중(kg)을 입력하세요:",
            min_value=80.0, value=80.0, step=1.0,
            help="이 체중을 기준으로 출하 가능 개체 수와 예측 날짜를 계산합니다."
        )

        if not pig_log_df_filtered.empty and predictor is not None:
            # (80kg 버그 수정 코드가 적용된) 유효한 체중 데이터만 필터링
            logs_with_weights = (
                pig_log_df_filtered.dropna(subset=["weight_kg"])
                if "weight_kg" in pig_log_df_filtered.columns else pd.DataFrame()
            )

            if not logs_with_weights.empty:
                latest_weights = logs_with_weights.loc[
                    logs_with_weights.groupby("pig_id")["timestamp"].idxmax()
                ]
                ship_ready_now = latest_weights[latest_weights["weight_kg"] >= target_weight]

                c1, c2 = st.columns(2)
                c1.metric(f"현재 {target_weight}kg 이상 (출하 가능)", f"{len(ship_ready_now)} 마리")

                # 'Mock' 대신 '예측 대기'로 변경
                pigs_below = latest_weights[latest_weights["weight_kg"] < target_weight]
                c2.metric("출하 예측 대상 (80kg 미만)", f"{len(pigs_below)} 마리")
                st.divider()

                st.subheader(f"🐷 {target_weight}kg 도달 날짜 예측 (AI 통계 기반)")

                if not pigs_below.empty:

                    # 2. 예측 로직 시작
                    results = []
                    today = pd.Timestamp.now()

                    for _, rep_pig in pigs_below.iterrows():
                        current_weight = rep_pig["weight_kg"]
                        pig_id = rep_pig["pig_id"]

                        # 이 돼지의 전체 로그 (체중, 날짜)
                        pig_data_hist = logs_with_weights[logs_with_weights['pig_id'] == pig_id].sort_values(
                            'timestamp')

                        # 3. 's2_predict.py'와 동일하게 '최근 7일' 증체율 계산
                        if len(pig_data_hist) >= 7:
                            recent_data = pig_data_hist.tail(7)
                        else:
                            recent_data = pig_data_hist

                        if len(recent_data) > 1:
                            weight_diff = recent_data['weight_kg'].iloc[-1] - recent_data['weight_kg'].iloc[0]
                            days_diff = (recent_data['timestamp'].iloc[-1] - recent_data['timestamp'].iloc[0]).days
                            recent_daily_gain = weight_diff / days_diff if days_diff > 0 else 0.6
                        else:
                            recent_daily_gain = 0.6  # (데이터가 1개면 기본값 0.6)

                        recent_daily_gain = max(0.2, recent_daily_gain)  # (음수 방지)

                        # 4. 예상 일령 계산 (단순화)
                        estimated_age = 30 + (
                                    pig_data_hist['timestamp'].iloc[-1] - pig_data_hist['timestamp'].iloc[0]).days

                        # 5. 예측 함수 호출
                        days_needed = predictor.predict_shipment_days(
                            current_weight,
                            estimated_age,
                            recent_daily_gain,
                            target_weight
                        )

                        predicted_date = today + pd.Timedelta(days=days_needed)

                        results.append({
                            '돼지 ID': pig_id,
                            '현재 체중(kg)': round(current_weight, 1),
                            '최근 증체율(kg/일)': round(recent_daily_gain, 3),
                            '남은 일수(일)': int(days_needed),
                            '예상 출하 날짜': predicted_date.strftime('%Y-%m-%d')
                        })

                    # 6. 예측 결과 테이블(DataFrame) 생성
                    result_df = pd.DataFrame(results).sort_values('남은 일수(일)')

                    fastest_pig = result_df.iloc[0]
                    st.metric(
                        f"가장 빠른 예상 출하일 (ID: {fastest_pig['돼지 ID']})",
                        f"{fastest_pig['예상 출하 날짜']}",
                        f"{fastest_pig['남은 일수(일)']}일 남음"
                    )

                    with st.expander("전체 개체별 예상 출하일 보기 (빠른 순)"):
                        st.dataframe(result_df.set_index('돼지 ID'), width='stretch')

                else:
                    if not ship_ready_now.empty:
                        st.success(f"모든 개체가 이미 목표 체중({target_weight}kg) 이상입니다.")
                    else:
                        st.info("분석할 유효한 체중 데이터가 없습니다.")
            else:
                st.warning("이 챔버에는 현재 유효한 체중 데이터가 없습니다.")
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
