"""
Step 2: 신규 돼지 예측 시스템 (수정본)
- Step 1에서 학습한 패턴 로드
- 정상적인 성장률로 2달치 랜덤 더미 데이터 생성
- 신규 돼지 출하 시점 예측
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
import os
import warnings

warnings.filterwarnings('ignore')

target_weight = 85

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
        self.target_weight = target_weight
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

    def predict_shipment_days(self, current_weight, current_age, recent_daily_gain):
        """
        출하까지 남은 일수 예측 (개선된 버전)
        """
        if current_weight >= self.target_weight:
            return 0

        remaining_weight = self.target_weight - current_weight

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


def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("🐷 돼지 출하 예측 시스템 - Step 2: 예측 (수정본)")
    print("=" * 80)

    # 1. 더미 데이터 생성
    generator = DummyDataGenerator(pattern_file='./growth_patterns.json')
    dummy_data = generator.generate_dummy_dataset(n_pigs=10, n_days=60,
                                                  output_path='./step2_dummy_data_fixed.csv')

    # 2. 예측 수행
    predictor = PatternBasedPredictor(target_weight=target_weight)
    results = predictor.analyze_new_pigs('./step2_dummy_data_fixed.csv')

    # 3. 결과 저장
    results.to_csv('./step2_prediction_results_fixed.csv', index=False, encoding='utf-8-sig')
    print(f"\n✓ 예측 결과 저장: ./step2_prediction_results_fixed.csv")

    # 4. 시각화
    predictor.visualize_predictions(dummy_data, results)

    print("\n" + "=" * 80)
    print("✅ Step 2 완료!")
    print("=" * 80)
    print("\n생성된 파일:")
    print("  1. step2_dummy_data_fixed.csv - 2달치 더미 데이터")
    print("  2. step2_prediction_results_fixed.csv - 예측 결과")
    print("  3. step2_predictions_fixed.png - 시각화 그래프")

    # 5. 상세 결과 출력
    print("\n" + "=" * 80)
    print("📋 상세 예측 결과")
    print("=" * 80)

    for _, pig in results.iterrows():
        print(f"\n🐷 Pig {pig['pig_id']}:")
        print(f"   현재 체중: {pig['current_weight']:.1f}kg (시작: {pig['start_weight']:.1f}kg)")
        print(f"   총 증체량: {pig['total_gain']:.1f}kg ({pig['current_day']}일간)")
        print(f"   평균 일일증체: {pig['avg_daily_gain']:.3f}kg/일")
        print(f"   최근 일일증체: {pig['recent_daily_gain']:.3f}kg/일")

        if pig['days_to_shipment']:
            print(f"   🎯 출하 예상: {pig['days_to_shipment']:.0f}일 후")
            print(f"   예상 출하일령: {pig['total_days_to_market']:.0f}일령")
            print(f"   예상 출하날짜: {pig['predicted_shipment_date']}")
        else:
            print(f"   ⚠️ 상태: {pig['status']}")


if __name__ == "__main__":
    main()